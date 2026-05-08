from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()


@app.post("/chat")
async def chat(request: dict):

    print("REQUEST RECEIVED")

    return JSONResponse(
        content={
            "version": "1.0",
            "response": {
                "outputSpeech": {
                    "type": "PlainText",
                    "text": "Hello Yogya. Ask me anything."
                },
                "shouldEndSession": False
            }
        },
        media_type="application/json"
    )
