from pinecone import PineconeAsyncio
import os
import asyncio

async def main():
    api_key = os.getenv("PINECONE_API_KEY", "dummy")
    pc = PineconeAsyncio(api_key=api_key)
    print(f"Attributes of PineconeAsyncio: {dir(pc)}")

if __name__ == "__main__":
    asyncio.run(main())