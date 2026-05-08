from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

@app.post("/chat")
async def chat(request: dict):

    try:

        request_type = request["request"]["type"]

        # =========================
        # Launch Request
        # =========================

        if request_type == "LaunchRequest":

            return JSONResponse(
                content={
                    "version": "1.0",
                    "response": {
                        "outputSpeech": {
                            "type": "PlainText",
                            "text": (
                                "Hello Yogya. "
                                "Ask me anything."
                            )
                        },
                        "shouldEndSession": False
                    }
                }
            )

        # =========================
        # Intent Request
        # =========================

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
                                    "You are a concise "
                                    "Alexa voice assistant."
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

                return JSONResponse(
                    content={
                        "version": "1.0",
                        "response": {
                            "outputSpeech": {
                                "type": "PlainText",
                                "text": answer
                            },
                            "shouldEndSession": False
                        }
                    }
                )

        # =========================
        # Fallback
        # =========================

        return JSONResponse(
            content={
                "version": "1.0",
                "response": {
                    "outputSpeech": {
                        "type": "PlainText",
                        "text": (
                            "I did not understand that."
                        )
                    },
                    "shouldEndSession": False
                }
            }
        )

    except Exception as e:

        print(e)

        return JSONResponse(
            content={
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
        )
