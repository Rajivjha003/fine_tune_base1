import logging
from core.system_init import SystemInitializer
import asyncio

logging.basicConfig(level=logging.DEBUG)

async def test():
    init = SystemInitializer()
    print("Checking GPU...")
    init.check_gpu()
    print("Checking configs...")
    init.check_configs()
    print("Ensuring dirs...")
    init.ensure_directories()
    
    print("Running ollama check...")
    await init.check_ollama()
    print("Running mlflow check...")
    await init.check_mlflow()
    print("Running redis check...")
    await init.check_redis()
    print("Done!")

if __name__ == "__main__":
    asyncio.run(test())
