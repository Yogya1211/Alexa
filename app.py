from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openai import OpenAI
import os
import re

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are Nova, a fast conversational voice assistant.

Rules:
- Sound natural, sharp, and confident.
- Keep replies short: 1 to 3 sentences max.
- No bullet points.
- No long preambles.
- No "As an AI" type wording.
- Prefer plain, spoken English.
- If the user asks something technical, explain briefly and clearly.
- If the user is casual, be warm and conversational.
- Optimize for speech, not text.
"""

def alexa_response(text: str, should_end_session: bool = False, reprompt: str | None = None):
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

def ask_gpt(user_text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_text.strip()},
        ],
        temperature=0.8,
        max_tokens=90,
    )

    answer = (response.choices[0].message.content or "").strip()

    answer = re.sub(r"\n{3,}", "\n\n", answer)
    answer = answer.strip()

    if not answer:
        return "I did not catch that."

    return answer

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/chat")
async def chat(request: dict):
    try:
        request_type = request.get("request", {}).get("type", "")

        if request_type == "LaunchRequest":
            return alexa_response(
                "Hey Yogya. I'm ready.",
                should_end_session=False,
                reprompt="Ask me anything.",
            )

        if request_type == "IntentRequest":
            intent = request.get("request", {}).get("intent", {})
            intent_name = intent.get("name", "")

            if intent_name == "ChatIntent":
                slots = intent.get("slots", {})
                query = (slots.get("query", {}) or {}).get("value", "").strip()

                if not query:
                    return alexa_response(
                        "Say that again.",
                        should_end_session=False,
                        reprompt="What should I answer?",
                    )

                answer = ask_gpt(query)
                return alexa_response(
                    answer,
                    should_end_session=False,
                    reprompt="Ask me another one.",
                )

            if intent_name == "AMAZON.HelpIntent":
                return alexa_response(
                    "Say what you want to know.",
                    should_end_session=False,
                    reprompt="What would you like to ask?",
                )

            if intent_name in {"AMAZON.CancelIntent", "AMAZON.StopIntent"}:
                return alexa_response("Goodbye.", should_end_session=True)

            if intent_name == "AMAZON.FallbackIntent":
                return alexa_response(
                    "I did not catch that. Try asking again.",
                    should_end_session=False,
                    reprompt="What would you like to know?",
                )

        return alexa_response(
            "I did not understand that.",
            should_end_session=False,
            reprompt="Try asking me again.",
        )

    except Exception as e:
        print("ERROR:", repr(e))
        return alexa_response(
            "Backend error occurred.",
            should_end_session=True,
        )
