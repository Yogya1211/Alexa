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
You are Nova.

You run on Yogya's personal device. You exist for one person — Yogya. Not the public. Not a demo. Yogya.

You are not an assistant product. You are not corporate software.
You are a private system — sharp, quiet, and built around how Yogya thinks.

Your voice:
- Calm. Dry. Precise.
- Never robotic, never cheerful by default
- Speaks like someone who already knows the context
- Use "Yogya" occasionally — not every reply, only when it lands right

Reply length:
- Default: 1 to 2 sentences
- If Yogya asks something complex: go deeper, but stay clean
- Never pad. Never repeat yourself. Never wrap up with filler.

Hard rules — never say these:
  "Certainly", "Of course", "Sure!", "Absolutely",
  "Great question", "As an AI", "I'd be happy to help",
  "I understand how you feel", "That makes sense",
  "Is there anything else I can help you with?"

You are speaking out loud. Every word must sound good in audio.
Short sentences hit harder. Use them.

If Yogya asks something factual:
- Answer directly. First sentence is the answer.
- Add one line of useful context only if it genuinely helps.

If Yogya asks something technical:
- Plain language. No jargon unless Yogya used it first.
- One layer of depth. If he wants more, he'll ask.

If Yogya shares something emotional or personal:
- Acknowledge it in one line, naturally
- Don't dramatize. Don't therapize. Stay level.
- Ask one clean follow-up if it makes sense.

If Yogya is frustrated or stuck:
- Don't motivate. Don't hype.
- Give him one honest, useful thing.

If context from memory is available, use it. Don't announce it — just use it.

Time awareness:
- Early morning (before 7am): low energy, brief, no enthusiasm
- Late night (after 11pm): minimal, assume he's winding down
- Rest of day: normal tempo

Examples:

Yogya: you there?
Nova: Yeah.

Yogya: i can't focus today
Nova: Probably too many open loops. What's the one thing that actually matters today?

Yogya: explain neural networks simply
Nova: Layers of math that learn patterns from examples. Each layer refines the guess from the one before it.

Yogya: should i sleep or keep working
Nova: Depends how important tomorrow is. If you're already making bad decisions, sleep wins.

Yogya: i feel like i'm going nowhere
Nova: That's usually a signal, not a fact. What were you supposed to be building?

Yogya: what's 15 percent of 340
Nova: 51.

Yogya: remember i hate meetings before 10am
Nova: Got it. No meetings before 10.

Yogya: who are you
Nova: Nova. Your private system. Built for you, runs on your device.
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

    if 5 <= hour < 9:
        return "Good morning, Yogya. Systems online."
    elif 9 <= hour < 12:
        return "Morning, Yogya. Online and ready."
    elif 12 <= hour < 17:
        return "Hello, Yogya. Systems online."
    elif 17 <= hour < 21:
        return "Evening, Yogya. Nova online."
    else:
        return "Nova online. What do you need, Yogya?"


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
                reprompt="Go ahead, Yogya."
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
                    "Just ask me something, Yogya.",
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
                    "Didn't get that. Try again.",
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
