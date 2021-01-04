import uvicorn
from fastapi import FastAPI
import asyncio

checking_result = {
    "lg_result": [],
    "additional_checking_result": {
        "aux" : [],
        "verbs":[],
        "usage": []
    }
}

app = FastAPI()

@app.get("/check_grammar")
async def check_grammar():
    await asyncio.sleep(5)
    return checking_result

if __name__ == "__main__":
    uvicorn.run("nlp:app", reload=True)