import requests
import asyncio
import aiohttp
from datetime import  datetime

nlp_service_url = "http://localhost:8000/check_grammar"

def sync_resuest(text= None):
    res = requests.get(url=nlp_service_url)
    if res.status_code == 200:
        print(res.text)


async def check_text(text=None):
    async with aiohttp.ClientSession() as session:
        async with session.get(nlp_service_url) as response:

            print("Status:", response.status)
            print("Content-type:", response.headers['content-type'])

            html = await response.text()
            print("Body:", html, "...")

## SYNC
# start = datetime.now()
#
# sync_resuest()
#
# print(f"Sync: {datetime.now() - start}")



## ASYNC
async def m(n):
    task = asyncio.create_task(check_text())
    await asyncio.gather(task)

start = datetime.now()
loop = asyncio.get_event_loop()
loop.run_until_complete(m(10))
print(f"Async: {datetime.now() - start}")