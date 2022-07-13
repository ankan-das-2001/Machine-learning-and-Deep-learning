import asyncio
import time

async def waiter(n):
    await asyncio.sleep(n)
    print(f"Waited {n} seconds")

async def main():
    task1 = asyncio.create_task(waiter(2))
    task2 = asyncio.create_task(waiter(3))
    task3 = asyncio.create_task(waiter(1))
    task4 = asyncio.create_task(waiter(2))

    print(time.strftime("%X"))
    await task1
    await task2
    await task3
    await task4
    print(time.strftime("%X"))

if __name__ == "__main__":
    asyncio.run(main())