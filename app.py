from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise voice assistant."
                )
            },
            {
                "role": "user",
                "content": query.text
            }
        ],
        max_completion_tokens=120
    )

    answer = (
        response.choices[0]
        .message.content
    )

    return {
        "reply": answer
    }
