from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openai import OpenAI
import os
import re

app = FastAPI()

# =========================================
# OpenAI Client
# =========================================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# =========================================
# Personality Prompt
# =========================================

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

You are SPEAKING, not writing.

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
"""

# =========================================
# Alexa Response Helper
# =========================================

def alexa_response(
    text: str,
    should_end_session: bool = False,
    reprompt: str | None = None
):

    payload = {
        "version": "1.0",
        "response": {
            "outputSpeech": {
                "type": "PlainText",
                "text": text[:800]
            },
            "shouldEndSession": should_end_session
        }
    }

    if reprompt and not should_end_session:

        payload["response"]["reprompt"] = {
            "outputSpeech": {
                "type": "PlainText",
                "text": reprompt[:300]
            }
        }

    return JSONResponse(content=payload)

# =========================================
# GPT Function
# =========================================

def ask_gpt(user_text: str) -> str:

    response = client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.strip()
            },
            {
                "role": "user",
                "content": user_text.strip()
            }
        ],
        temperature=0.95,
        max_completion_tokens=60
    )

    answer = (
        response.choices[0]
        .message.content
        or ""
    ).strip()

    # Clean formatting for speech
    answer = re.sub(r"\s+", " ", answer)

    if not answer:
        return "Say that again."

    return answer

# =========================================
# Root Route
# =========================================

@app.get("/")
async def root():

    return {
        "status": "online"
    }

# =========================================
# Alexa Endpoint
# =========================================

@app.post("/chat")
async def chat(request: dict):

    try:

        print("\n========== REQUEST ==========")
        print(request)

        request_type = (
            request.get("request", {})
            .get("type", "")
        )

        # =====================================
        # Launch Request
        # =====================================

        if request_type == "LaunchRequest":

            return alexa_response(
                "Yeah?",
                should_end_session=False,
                reprompt="Go ahead."
            )

        # =====================================
        # Intent Request
        # =====================================

        if request_type == "IntentRequest":

            intent = (
                request.get("request", {})
                .get("intent", {})
            )

            intent_name = (
                intent.get("name", "")
            )

            print("INTENT:", intent_name)

            # =====================================
            # Chat Intent
            # =====================================

            if intent_name == "ChatIntent":

                slots = intent.get("slots", {})

                query = (
                    (slots.get("query", {}) or {})
                    .get("value", "")
                    .strip()
                )

                print("QUERY:", query)

                if not query:

                    return alexa_response(
                        "Didn't catch that.",
                        should_end_session=False,
                        reprompt="Say it again."
                    )

                answer = ask_gpt(query)

                print("ANSWER:", answer)

                return alexa_response(
                    answer,
                    should_end_session=False,
                    reprompt="Anything else?"
                )

            # =====================================
            # Help Intent
            # =====================================

            if intent_name == "AMAZON.HelpIntent":

                return alexa_response(
                    "Just ask something.",
                    should_end_session=False,
                    reprompt="Go ahead."
                )

            # =====================================
            # Stop / Cancel
            # =====================================

            if intent_name in {
                "AMAZON.StopIntent",
                "AMAZON.CancelIntent"
            }:

                return alexa_response(
                    "Alright.",
                    should_end_session=True
                )

            # =====================================
            # Fallback
            # =====================================

            if intent_name == "AMAZON.FallbackIntent":

                return alexa_response(
                    "Not sure what you meant.",
                    should_end_session=False,
                    reprompt="Try saying it differently."
                )

        # =====================================
        # Unknown Request
        # =====================================

        return alexa_response(
            "I didn't understand that.",
            should_end_session=False,
            reprompt="Try again."
        )

    except Exception as e:

        print("\n========== ERROR ==========")
        print(repr(e))

        return alexa_response(
            "Something broke on my side.",
            should_end_session=True
        )
