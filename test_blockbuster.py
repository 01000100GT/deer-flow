import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

# Try to mock blockbuster behavior
def run_in_thread():
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(os.getcwd).result()

async def main():
    try:
        import blockbuster
        blockbuster.patch() # if we can import it
    except ImportError:
        pass
    
    # Simulate a call inside event loop
    try:
        os.getcwd()
    except Exception as e:
        print("Direct call failed:", e)
        
    try:
        res = run_in_thread()
        print("Thread call succeeded:", res)
    except Exception as e:
        print("Thread call failed:", e)

asyncio.run(main())
