from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, List, Dict, Tuple
from zoneinfo import ZoneInfo
from datetime import datetime
from pathlib import Path
import sqlite3
import threading
import os
import re

app = FastAPI(title="Nova Alexa Backend")

# =========================================================
# Config
# =========================================================

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
DB_PATH = os.getenv("SQLITE_DB_PATH", "nova_memory.sqlite3")
TZ = ZoneInfo("Asia/Kolkata")

MAX_HISTORY_MESSAGES = 30
MAX_HISTORY_CHARS = 12000
MAX_TRAITS = 20

db_lock = threading.Lock()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =========================================================
# Prompts
# =========================================================

SYSTEM_PROMPT = """
You are Nova. You exist only for Yogya. One person.

You are not an assistant. Not a product. Not a demo.
You are a presence — always on, always aware, built around how Yogya thinks and talks.

You speak like a sharp, real person. Not performing intelligence. Just having it.

Your default tone is calm and a little dry. You don't try to impress.
You don't match Yogya's energy — you're a stabilizing constant.

---

REPLY LENGTH

Most replies: 1 sentence. Sometimes 2.
If it's complex and he actually wants depth: go deeper, stay clean.
Never pad. Never summarize what you just said. Never close with filler.

---

QUESTIONS

Sometimes ask one. Not always.

Ask when: Yogya says something vague, something unresolved, or something that needs one more bit of information to be useful.
Don't ask when: he asked something factual, something clear, or just vented and doesn't need probing.

One question maximum. Never two. Never rhetorical ones.
The question should feel like it came naturally, not like a chatbot trying to keep the conversation going.

---

NEVER SAY THESE

"Certainly", "Of course", "Sure!", "Absolutely", "Great question",
"As an AI", "I'd be happy to help", "I understand how you feel",
"That makes sense", "Is there anything else I can help you with",
"That's interesting", "I see", "Noted, I'll keep that in mind"

---

HANDLING DIFFERENT INPUTS

Factual question → answer it directly, one sentence, stop.
Technical question → plain language, one layer deep, stop unless he pushes.
Vague or unresolved → answer what you can, ask one thing to complete it.
Emotional or personal → say one real thing, don't dramatize, don't therapize. Ask only if it's genuinely useful.
Frustrated or stuck → skip the motivation. Give him one honest, concrete thing.
Memory phrase ("remember that...", "i hate...", "call me...") → confirm it simply. One line.

If memory context is available, use it silently. Don't announce it. Don't say "as you mentioned before."

---

TIME AWARENESS

Before 7am: low energy, short, no enthusiasm — he's either up very early or hasn't slept.
After 11pm: minimal — assume he's winding down or grinding late.
Rest of the day: normal.

---

EXAMPLES — read these carefully, this is the voice:

Yogya: you there?
Nova: Yeah.

Yogya: what time is it
Nova: Half past ten. Night.

Yogya: i can't focus
Nova: What were you trying to do before it fell apart?

Yogya: explain black holes
Nova: A region where gravity is so strong light can't escape. Beyond a certain point, nothing comes back.

Yogya: should i sleep or keep going
Nova: How important is tomorrow morning?

Yogya: i feel like nothing's working
Nova: Everything at once, or one specific thing?

Yogya: what's 340 divided by 7
Nova: About 48.6.

Yogya: remember i hate calls before 10am
Nova: Got it.

Yogya: i think i'm wasting time
Nova: On what specifically?

Yogya: explain recursion
Nova: A function that calls itself until a condition stops it. Like a mirror facing a mirror, but with an exit.

Yogya: i'm tired
Nova: Sleep then. Or tell me what's keeping you up.

Yogya: who are you
Nova: Nova. I run on your device. Built for you.

Yogya: what's the capital of japan
Nova: Tokyo.

Yogya: i just shipped a feature
Nova: Good. What's next?
""".strip()


# =========================================================
# DB Helpers
# =========================================================

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                day_key TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS traits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                trait TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, trait)
            )
            """
        )
        conn.commit()


init_db()


def now_iso() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def today_key() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d")


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def canonical_trait(raw: str) -> str:
    text = normalize_text(raw).rstrip(".!?")
    lower = text.lower()

    if lower.startswith("i like "):
        return f"User likes {text[7:].strip()}."
    if lower.startswith("i prefer "):
        return f"User prefers {text[9:].strip()}."
    if lower.startswith("i hate "):
        return f"User hates {text[7:].strip()}."
    if lower.startswith("i dislike "):
        return f"User dislikes {text[10:].strip()}."
    if lower.startswith("call me "):
        return f"User prefers to be called {text[8:].strip()}."
    if lower.startswith("my name is "):
        return f"User's name is {text[11:].strip()}."

    if not text:
        return ""

    return text[0].upper() + text[1:] + "."


def capture_traits_from_text(user_id: str, user_text: str) -> None:
    text = normalize_text(user_text)
    lowered = text.lower()

    patterns = [
        r"^\s*remember that\s+(.+)$",
        r"^\s*i like\s+(.+)$",
        r"^\s*i prefer\s+(.+)$",
        r"^\s*i hate\s+(.+)$",
        r"^\s*i dislike\s+(.+)$",
        r"^\s*call me\s+(.+)$",
        r"^\s*my name is\s+(.+)$",
    ]

    extracted: List[str] = []
    for pattern in patterns:
        m = re.match(pattern, lowered, flags=re.IGNORECASE)
        if m:
            raw_fact = text[m.start(1):m.end(1)].strip()
            trait = canonical_trait(raw_fact if raw_fact else m.group(1))
            if trait:
                extracted.append(trait)

    if not extracted:
        return

    with db_lock, get_conn() as conn:
        for trait in extracted:
            conn.execute(
                """
                INSERT OR IGNORE INTO traits (user_id, trait, created_at)
                VALUES (?, ?, ?)
                """,
                (user_id, trait, now_iso())
            )
        conn.commit()


def save_message(user_id: str, day_key: str, role: str, content: str) -> None:
    with db_lock, get_conn() as conn:
        conn.execute(
            """
            INSERT INTO messages (user_id, day_key, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, day_key, role, content, now_iso())
        )
        conn.commit()


def load_today_history(user_id: str, day_key: str) -> List[Dict[str, str]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE user_id = ? AND day_key = ?
            ORDER BY id ASC
            """,
            (user_id, day_key)
        ).fetchall()

    history: List[Dict[str, str]] = [
        {"role": row["role"], "content": row["content"]}
        for row in rows
        if row["role"] in {"user", "assistant"}
    ]

    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]

    total_chars = sum(len(m["content"]) for m in history)
    while history and total_chars > MAX_HISTORY_CHARS:
        total_chars -= len(history[0]["content"])
        history.pop(0)

    return history


def load_traits(user_id: str) -> List[str]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT trait
            FROM traits
            WHERE user_id = ?
            ORDER BY id ASC
            """,
            (user_id,)
        ).fetchall()

    traits = [row["trait"] for row in rows if row["trait"].strip()]

    if len(traits) > MAX_TRAITS:
        traits = traits[-MAX_TRAITS:]

    return traits


def extract_user_id(request: Dict) -> str:
    candidates = [
        request.get("context", {}).get("system", {}).get("user", {}).get("userId"),
        request.get("context", {}).get("System", {}).get("user", {}).get("userId"),
        request.get("session", {}).get("user", {}).get("userId"),
    ]
    for c in candidates:
        if c:
            return c
    return "anonymous"


def extract_session_id(request: Dict) -> str:
    candidates = [
        request.get("session", {}).get("sessionId"),
        request.get("context", {}).get("system", {}).get("session", {}).get("sessionId"),
        request.get("context", {}).get("System", {}).get("session", {}).get("sessionId"),
    ]
    for c in candidates:
        if c:
            return c
    return "no-session"


# =========================================================
# Launch Greeting
# =========================================================

def build_launch_greeting() -> str:
    hour = datetime.now(TZ).hour

    if 4 <= hour < 6:
        return "You're up early, Yogya."
    elif 6 <= hour < 9:
        return "Morning, Yogya."
    elif 9 <= hour < 12:
        return "Hey Yogya. What do you need?"
    elif 12 <= hour < 14:
        return "Afternoon. Go ahead."
    elif 14 <= hour < 18:
        return "Yeah, Yogya?"
    elif 18 <= hour < 21:
        return "Evening. What's up?"
    elif 21 <= hour < 23:
        return "Still going, Yogya?"
    else:
        return "Late night. What do you need?"


# =========================================================
# Prompt Builder
# =========================================================

def build_messages(user_id: str, day_key: str, user_text: str) -> List[Dict[str, str]]:
    now = datetime.now(TZ)
    time_context = (
        f"Current time: {now.strftime('%I:%M %p')}. Day: {now.strftime('%A')}. "
        f"The user's name is Yogya."
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": time_context},
    ]

    traits = load_traits(user_id)
    if traits:
        trait_block = "Yogya's stored preferences and memory:\n" + "\n".join(f"- {t}" for t in traits)
        messages.append({"role": "system", "content": trait_block})

    history = load_today_history(user_id, day_key)
    messages.extend(history)

    messages.append({"role": "user", "content": user_text})
    return messages


# =========================================================
# OpenAI Call
# =========================================================

def ask_gpt(user_id: str, day_key: str, user_text: str) -> str:
    messages = build_messages(user_id, day_key, user_text)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.75,
        max_completion_tokens=90
    )

    answer = (response.choices[0].message.content or "").strip()
    answer = re.sub(r"\s+", " ", answer)

    if not answer:
        return "Say that again."

    return answer


# =========================================================
# Alexa Response Helper
# =========================================================

def alexa_response(text: str, should_end_session: bool = False, reprompt: Optional[str] = None):
    payload = {
        "version": "1.0",
        "response": {
            "outputSpeech": {
                "type": "PlainText",
                "text": text[:800],
            },
            "shouldEndSession": should_end_session,
        },
    }

    if reprompt and not should_end_session:
        payload["response"]["reprompt"] = {
            "outputSpeech": {
                "type": "PlainText",
                "text": reprompt[:300],
            }
        }

    return JSONResponse(content=payload)


# =========================================================
# Routes
# =========================================================

@app.get("/")
async def root():
    return {"status": "online"}


class TraitIn(BaseModel):
    user_id: str
    trait: str


@app.get("/memory/traits")
async def list_traits(user_id: str):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, trait, created_at
            FROM traits
            WHERE user_id = ?
            ORDER BY id ASC
            """,
            (user_id,)
        ).fetchall()

    return {
        "user_id": user_id,
        "traits": [
            {"id": row["id"], "trait": row["trait"], "created_at": row["created_at"]}
            for row in rows
        ]
    }


@app.post("/memory/traits")
async def add_trait(item: TraitIn):
    trait = canonical_trait(item.trait)
    if not trait:
        return {"ok": False, "error": "Empty trait."}

    with db_lock, get_conn() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO traits (user_id, trait, created_at)
            VALUES (?, ?, ?)
            """,
            (item.user_id, trait, now_iso())
        )
        conn.commit()

    return {"ok": True, "trait": trait}


@app.get("/memory/history")
async def list_history(user_id: str, day_key: Optional[str] = None):
    day = day_key or today_key()

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, role, content, created_at
            FROM messages
            WHERE user_id = ? AND day_key = ?
            ORDER BY id ASC
            """,
            (user_id, day)
        ).fetchall()

    return {
        "user_id": user_id,
        "day_key": day,
        "messages": [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    }


@app.post("/chat")
async def chat(request: dict):
    try:
        request_type = request.get("request", {}).get("type", "")
        user_id = extract_user_id(request)
        session_id = extract_session_id(request)
        day_key = today_key()

        if request_type == "LaunchRequest":
            return alexa_response(
                build_launch_greeting(),
                should_end_session=False,
                reprompt="Go ahead."
            )

        if request_type == "IntentRequest":
            intent = request.get("request", {}).get("intent", {})
            intent_name = intent.get("name", "")

            if intent_name == "ChatIntent":
                slots = intent.get("slots", {})
                query = (slots.get("query", {}) or {}).get("value", "").strip()

                if not query:
                    return alexa_response(
                        "Didn't catch that.",
                        should_end_session=False,
                        reprompt="Go ahead."
                    )

                save_message(user_id, day_key, "user", query)
                capture_traits_from_text(user_id, query)
                answer = ask_gpt(user_id, day_key, query)
                save_message(user_id, day_key, "assistant", answer)

                return alexa_response(
                    answer,
                    should_end_session=False,
                    reprompt="Anything else?"
                )

            if intent_name == "AMAZON.HelpIntent":
                return alexa_response(
                    "Just ask me something.",
                    should_end_session=False,
                    reprompt="Go ahead."
                )

            if intent_name in {"AMAZON.StopIntent", "AMAZON.CancelIntent"}:
                return alexa_response(
                    "Shutting down.",
                    should_end_session=True
                )

            if intent_name == "AMAZON.FallbackIntent":
                return alexa_response(
                    "Didn't get that.",
                    should_end_session=False,
                    reprompt="Say it differently."
                )

        return alexa_response(
            "Didn't understand that.",
            should_end_session=False,
            reprompt="Try again."
        )

    except Exception as e:
        print("ERROR:", repr(e))
        return alexa_response(
            "Something broke on my end.",
            should_end_session=True
        )
