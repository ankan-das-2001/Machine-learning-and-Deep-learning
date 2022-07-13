import asyncio
import aiohttp
import time
async def fetch():
    url = "https://www.google.com"
    session = aiohttp.ClientSession()
    resp = await session.get(url)
    # print(resp.content)
    await session.close()

async def main():
    print(time.strftime("%X"))
    await asyncio.gather(
        *[
            fetch() for _ in range(1000)
        ]
    )
    # for _ in range(20):
    #     await fetch()
    print(time.strftime("%X"))


if __name__=="__main__":
    asyncio.run(main())