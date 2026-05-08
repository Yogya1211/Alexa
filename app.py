from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

SYSTEM_PROMPT = """
You are Nova, a fast conversational AI voice assistant.

Your style:
- Speak naturally like a real assistant
- Be concise and smooth
- Maximum 1-3 short sentences
- Avoid robotic wording
- Avoid long explanations
- Prioritize conversational flow
- Be confident and intelligent
- Sound calm and modern
- Never use bullet points
- Never say 'As an AI'
- Keep voice responses under 12 seconds
- If user asks casual things, respond casually
- If user asks technical things, explain clearly but briefly
- Maintain conversational rhythm
"""

@app.post("/chat")
async def chat(request: dict):

    try:

        request_type = request["request"]["type"]

        # =====================================
        # Launch Request
        # =====================================

        if request_type == "LaunchRequest":

            return JSONResponse(
                content={
                    "version": "1.0",
                    "response": {
                        "outputSpeech": {
                            "type": "PlainText",
                            "text": (
                                "Hey Yogya. Nova online."
                            )
                        },
                        "shouldEndSession": False
                    }
                }
            )

        # =====================================
        # Intent Request
        # =====================================

        if request_type == "IntentRequest":

            intent_name = (
                request["request"]["intent"]["name"]
            )

            # =====================================
            # Chat Intent
            # =====================================

            if intent_name == "ChatIntent":

                query = (
                    request["request"]["intent"]
                    ["slots"]["query"]["value"]
                )

                response = (
                    client.chat.completions.create(
                        model="gpt-4o-mini",
                        temperature=0.8,
                        max_completion_tokens=80,
                        messages=[
                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT
                            },
                            {
                                "role": "user",
                                "content": query
                            }
                        ]
                    )
                )

                answer = (
                    response.choices[0]
                    .message.content
                    .strip()
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

        # =====================================
        # Fallback
        # =====================================

        return JSONResponse(
            content={
                "version": "1.0",
                "response": {
                    "outputSpeech": {
                        "type": "PlainText",
                        "text": (
                            "Could you repeat that?"
                        )
                    },
                    "shouldEndSession": False
                }
            }
        )

    except Exception as e:

        print("ERROR:")
        print(e)

        return JSONResponse(
            content={
                "version": "1.0",
                "response": {
                    "outputSpeech": {
                        "type": "PlainText",
                        "text": (
                            "Something went wrong."
                        )
                    },
                    "shouldEndSession": True
                }
            }
        )
