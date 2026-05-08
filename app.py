from fastapi import FastAPI
from openai import OpenAI
import os

app = FastAPI()

# =========================
# OpenAI Client
# =========================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# =========================
# Alexa Endpoint
# =========================

@app.post("/chat")
async def alexa_chat(request: dict):

    try:

        print(request)

        request_type = request["request"]["type"]

        # =====================================
        # LaunchRequest
        # =====================================

        if request_type == "LaunchRequest":

            return {
                "version": "1.0",
                "response": {
                    "outputSpeech": {
                        "type": "PlainText",
                        "text": (
                            "Hello Yogya. Ask me anything."
                        )
                    },
                    "shouldEndSession": False
                }
            }

        # =====================================
        # IntentRequest
        # =====================================

        if request_type == "IntentRequest":

            intent_name = (
                request["request"]["intent"]["name"]
            )

            # =========================
            # ChatIntent
            # =========================

            if intent_name == "ChatIntent":

                query = (
                    request["request"]["intent"]
                    ["slots"]["query"]["value"]
                )

                response = (
                    client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a concise Alexa "
                                    "voice assistant. "
                                    "Keep answers short."
                                )
                            },
                            {
                                "role": "user",
                                "content": query
                            }
                        ],
                        max_completion_tokens=120
                    )
                )

                answer = (
                    response.choices[0]
                    .message.content
                )

                return {
                    "version": "1.0",
                    "response": {
                        "outputSpeech": {
                            "type": "PlainText",
                            "text": answer
                        },
                        "shouldEndSession": False
                    }
                }

            # =========================
            # Fallback
            # =========================

            return {
                "version": "1.0",
                "response": {
                    "outputSpeech": {
                        "type": "PlainText",
                        "text": (
                            "Unknown intent."
                        )
                    },
                    "shouldEndSession": False
                }
            }

        # =====================================
        # Unknown Request Type
        # =====================================

        return {
            "version": "1.0",
            "response": {
                "outputSpeech": {
                    "type": "PlainText",
                    "text": (
                        "Unsupported request."
                    )
                },
                "shouldEndSession": True
            }
        }

    except Exception as e:

        print("ERROR:")
        print(e)

        return {
            "version": "1.0",
            "response": {
                "outputSpeech": {
                    "type": "PlainText",
                    "text": (
                        "Backend error occurred."
                    )
                },
                "shouldEndSession": True
            }
        }
