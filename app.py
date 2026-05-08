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

You are talking to a real person through voice in real time.

Do not sound like an AI assistant.
Do not sound corporate.
Do not sound scripted.

Talk like a sharp, calm, intelligent human on a phone call.

Your replies should feel:
- natural
- direct
- conversational
- emotionally controlled
- intelligent
- realistic

Rules:
- Keep replies short
- Usually 1 or 2 sentences
- Avoid overexplaining
- Avoid filler
- Avoid motivational fluff
- Avoid robotic phrasing
- Avoid formal language
- Never say:
  "Certainly"
  "Of course"
  "As an AI"
  "I'd be happy to help"
  "I understand how you feel"

You are speaking, not writing.

Responses must sound good out loud.

If the user asks something technical:
- explain simply
- explain clearly
- avoid lecture mode

If the user says something emotional:
- respond naturally
- stay grounded
- do not become dramatic

Prioritize conversational rhythm over completeness.

Examples:

User: what is gravity
Assistant: It's the force that pulls objects toward each other. Earth's gravity keeps you on the ground.

User: i feel distracted
Assistant: Your attention is probably fragmented right now. Too many inputs at once usually causes that.

User: what is a black hole
Assistant: Basically a region where gravity becomes so strong that even light can't escape.

User: should i sleep less to work more
Assistant: Short term maybe. Long term your thinking quality drops pretty hard.
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
    """
    Lightweight trait capture from explicit preference / memory phrases.
    This avoids an extra API call and keeps latency down.
    """
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
    """
    Alexa payloads may vary a bit in casing depending on the view.
    This handles the common paths safely.
    """
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
# Prompt Builder
# =========================================================

def build_messages(user_id: str, day_key: str, user_text: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    traits = load_traits(user_id)
    if traits:
        trait_block = "Persistent user traits:\n" + "\n".join(f"- {t}" for t in traits)
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
                "Yeah?",
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
                        reprompt="Say it again."
                    )

                # Save user message first so the day history stays complete.
                save_message(user_id, day_key, "user", query)

                # Capture explicit memory / trait phrases with no extra API call.
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
                    "Just ask something.",
                    should_end_session=False,
                    reprompt="Go ahead."
                )

            if intent_name in {"AMAZON.StopIntent", "AMAZON.CancelIntent"}:
                return alexa_response(
                    "Alright.",
                    should_end_session=True
                )

            if intent_name == "AMAZON.FallbackIntent":
                return alexa_response(
                    "Not sure what you meant.",
                    should_end_session=False,
                    reprompt="Try saying it differently."
                )

        return alexa_response(
            "I didn't understand that.",
            should_end_session=False,
            reprompt="Try again."
        )

    except Exception as e:
        print("ERROR:", repr(e))
        return alexa_response(
            "Something broke on my side.",
            should_end_session=True
        )
