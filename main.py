import asyncio
import pytz
from datetime import datetime
from telebot import start_bot
from scheduler import scheduler_loop
from portfolio import start_cache_refresh_loop

_log_count = 0

def log(msg: str):
    global _log_count
    _log_count += 1
    uk_time = datetime.now(pytz.timezone("Europe/London"))
    ts = uk_time.strftime("%H:%M:%S")
    print(f"[{ts}] [{_log_count}] {msg}")

async def main():
    log("🚀 Started portfolio tracker and scheduler")
    # Run Telegram bot + scheduler + cache refresh loop concurrently
    tasks = [
        asyncio.create_task(start_bot()),
        asyncio.create_task(scheduler_loop()),
        asyncio.create_task(start_cache_refresh_loop()),
    ]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        log("⚠️ Main tasks cancelled, shutting down gracefully")
    finally:
        log("🛑 Portfolio tracker and scheduler stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Received Ctrl+C — exiting cleanly")
