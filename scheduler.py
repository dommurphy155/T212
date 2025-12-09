import asyncio
import pytz
from datetime import datetime
from news import fetch_latest_news
from portfolio import fetch_all_data
import time
import os
# FIX: Import CHAT_ID from telebot for scheduled sends
from telebot import send_message, CHAT_ID 

# === CONFIG ===
TICKERS = ["NVDA", "AVGO", "PLTR", "AMD"]
# Define (hour, minute) for UK times — supports any minute
SCHEDULED_TIMES = [(10, 30), (14, 50), (18, 30), (14, 8)]
RATE_LIMIT_KEY_DELAY = 60  # 2 minutes between OpenAI calls per ticker

_current_key_index = 0

_log_count = 0

def log(msg: str):
    global _log_count
    _log_count += 1
    uk_time = datetime.now(pytz.timezone("Europe/London"))
    ts = uk_time.strftime("%H:%M:%S")
    print(f"[{ts}] [{_log_count}] {msg}")

async def send_news_cycle():
    """Send news summaries for each ticker with OpenAI key rotation."""
    from news import OPENAI_KEYS
    global _current_key_index

    for i, ticker in enumerate(TICKERS):
        # Rotate through OpenAI keys
        active_key = OPENAI_KEYS[_current_key_index % len(OPENAI_KEYS)]
        _current_key_index += 1
        os.environ["OPENAI_API_KEY"] = active_key

        log(f"📰 Fetching news for {ticker} with key #{_current_key_index}")

        try:
            headlines = await fetch_latest_news(ticker)
            if headlines:
                # FIX: Iterate through each news item and send it with the full inline keyboard 
                # (to ensure 'Read Article' and 'Follow Up' buttons are included, matching the manual flow)
                for item in headlines:
                    summary = item["summary"]
                    url = item.get("url")

                    # Generate the comprehensive keyboard (matching the fix in telebot.py)
                    keyboard = {
                        "inline_keyboard": [
                            # General Actions
                            [
                                {"text": "📊 Account Summary", "callback_data": "account_summary"},
                                {"text": "📰 Fetch News", "callback_data": "fetch_news"},
                            ],
                            # Contextual Actions (Read/Follow Up)
                            ([{"text": "🔗 Read Article", "url": url}] if url else []) + [{"text": "💭 Follow Up", "callback_data": f"followup_{ticker}"}],
                            # Home
                            [{"text": "🏡 Home", "callback_data": "home"}]
                        ]
                    }
                    
                    await send_message(
                        summary,
                        parse_mode="Markdown",
                        chat_id_override=CHAT_ID, # Use the global CHAT_ID for scheduled messages
                        inline_keyboard=keyboard
                    )

            else:
                log(f"ℹ️ No new headlines for {ticker}")
        except Exception as e:
            log(f"⚠️ News fetch failed for {ticker}: {e}")

        if i < len(TICKERS) - 1:
            log(f"⏳ Waiting {RATE_LIMIT_KEY_DELAY}s before next ticker")
            await asyncio.sleep(RATE_LIMIT_KEY_DELAY)

async def scheduler_loop():
    """Runs news updates at the specified UK times daily."""
    log("🕒 Started scheduler")
    last_run_time = None

    while True:
        uk_now = datetime.now(pytz.timezone("Europe/London"))
        current_hour = uk_now.hour
        current_min = uk_now.minute

        # Run only once per scheduled minute
        if (current_hour, current_min) in SCHEDULED_TIMES:
            if last_run_time != (current_hour, current_min):
                log(f"⏰ Running scheduled news cycle at {current_hour:02d}:{current_min:02d} UK")
                await send_news_cycle()
                last_run_time = (current_hour, current_min)
            await asyncio.sleep(60)  # avoid re-triggering same minute
        await asyncio.sleep(20)  # check every 20s
