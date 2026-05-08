from fastapi import FastAPI
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

@app.post("/chat")
async def chat(request: dict):

    try:

        query = (
            request["request"]["intent"]
            ["slots"]["query"]["value"]
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise Alexa voice assistant."
                    )
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            max_completion_tokens=100
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

    except Exception as e:

        print(e)

        return {
            "version": "1.0",
            "response": {
                "outputSpeech": {
                    "type": "PlainText",
                    "text": "Backend error occurred."
                },
                "shouldEndSession": True
            }
        }
