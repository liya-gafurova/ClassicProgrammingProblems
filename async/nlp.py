import hashlib
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastapi import status
import asyncio

from pydantic import BaseModel

checking_result = {
    "lg_result": [],
    "additional_checking_result": {
        "aux" : [],
        "verbs":[],
        "usage": []
    }
}

def _create_hash_for_text(text):
    current_time =  datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    hash_object = hashlib.md5(text.encode()+current_time.encode())
    return hash_object.hexdigest()

class ResponseTextIsAccepted(BaseModel):
    hash: str

app = FastAPI()

@app.get("/check_grammar")
async def check_grammar() -> ResponseTextIsAccepted:
    test_hash = _create_hash_for_text("hello")
    await asyncio.sleep(20)
    return ResponseTextIsAccepted(hash = test_hash), status.HTTP_200_OK

if __name__ == "__main__":
    uvicorn.run("nlp:app", reload=True)