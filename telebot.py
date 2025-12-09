import os
import asyncio
import pytz
from typing import List, Optional
from dotenv import load_dotenv
from datetime import datetime
import httpx
from portfolio import fetch_all_data, build_summary
from news import fetch_latest_news
from follow_up import generate_follow_up_response

# === LOAD ENV ===
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TOKEN or not CHAT_ID:
    raise RuntimeError("❌ Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env file")

BASE_URL = f"https://api.telegram.org/bot{TOKEN}"

# === HTTP CLIENT ===
http_client = httpx.AsyncClient(
    timeout=30,
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
)

# === GLOBAL STATE ===
awaiting_ticker = set()
awaiting_follow_up = {} 

_log_count = 0

def log(msg: str):
    global _log_count
    _log_count += 1
    uk_time = datetime.now(pytz.timezone("Europe/London"))
    ts = uk_time.strftime("%H:%M:%S")
    print(f"[{ts}] [{_log_count}] {msg}")

async def send_message(
    message: str,
    parse_mode: Optional[str] = None,
    chat_id_override: Optional[int] = None,
    news_prompt: bool = False,
    home_only: bool = False,
    inline_keyboard: Optional[dict] = None
) -> None:
    """Send Telegram message with optional keyboard."""
    log(f"📤 Sending message: {message[:30]}…")
    chat_id = chat_id_override or CHAT_ID
    url = f"{BASE_URL}/sendMessage"
    chunks: List[str] = [message[i:i + 4000] for i in range(0, len(message), 4000)]

    # === Keyboard Layout ===
    if inline_keyboard:
        keyboard = inline_keyboard
    elif home_only:
        keyboard = {"inline_keyboard": [[{"text": "🏡 Home", "callback_data": "home"}]]}
    elif news_prompt:
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "🚗 TSLA", "callback_data": "news_TSLA"},
                    {"text": "🧠 NVDA", "callback_data": "news_NVDA"},
                ],
                [
                    {"text": "⚙️ AVGO", "callback_data": "news_AVGO"},
                    {"text": "🛰️ PLTR", "callback_data": "news_PLTR"},
                ],
                [{"text": "🏡 Home", "callback_data": "home"}],
            ]
        }
    else:
        # Default menu: Account Summary, Fetch News, Home
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "📊 Account Summary", "callback_data": "account_summary"},
                    {"text": "📰 Fetch News", "callback_data": "fetch_news"},
                ],
                [{"text": "🏡 Home", "callback_data": "home"}],
            ]
        }

    data_base = {"chat_id": chat_id, "reply_markup": keyboard}
    if parse_mode:
        data_base["parse_mode"] = parse_mode

    for chunk in chunks:
        data = dict(data_base)
        data["text"] = chunk
        try:
            resp = await http_client.post(url, json=data)
            if resp.status_code != 200:
                log(f"⚠️ Telegram API error {resp.status_code}")
        except httpx.RequestError as e:
            log(f"⚠️ Network error sending message: {e}")

async def send_inline_keyboard(chat_id: str) -> None:
    """Send home keyboard."""
    log(f"🖼️ Sending inline keyboard to chat {chat_id}")
    url = f"{BASE_URL}/sendMessage"
    keyboard = {
        "inline_keyboard": [
            [
                {"text": "📊 Account Summary", "callback_data": "account_summary"},
                {"text": "📰 Fetch News", "callback_data": "fetch_news"},
            ],
            [{"text": "🏡 Home", "callback_data": "home"}],
        ]
    }
    message = (
        "🏦 *Welcome to your Trading212 Portfolio!*\n"
        "Choose what you’d like to see below 👇"
    )
    await http_client.post(url, json={
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
        "reply_markup": keyboard
    })

async def handle_callback(callback_query: dict) -> None:
    """Handle inline button callbacks."""
    data = callback_query.get("data")
    chat_id = callback_query["message"]["chat"]["id"]
    callback_id = callback_query.get("id")
    log(f"🔄 Processing callback: {data}")

    await http_client.post(
        f"{BASE_URL}/answerCallbackQuery",
        data={"callback_query_id": callback_id, "text": "Processing..."},
    )

    if data == "account_summary":
        await send_message("⏳ Fetching your portfolio summary...", chat_id_override=chat_id, home_only=True)
        # FIX: fetch_all_data now returns 3 values (data, net_deposits, missing).
        # Removed the redundant net_investments unpacking.
        data_out, net_deposits, missing = await fetch_all_data()
        # build_summary is called with 3 positional arguments (net_investments defaults to None)
        summary = await build_summary(data_out, net_deposits, missing)
        await send_message(summary, chat_id_override=chat_id)

    elif data == "fetch_news":
        awaiting_ticker.add(chat_id)
        await send_message(
            "📰 What asset would you like news for?\n\n"
            "Please choose a ticker below 👇\n"
            "Or just tell me a ticker and I'll fetch it for you!",
                           chat_id_override=chat_id, news_prompt=True)

    elif data.startswith("news_"):
        ticker = data.split("_")[1]
        
        # FIX: Clear the awaiting_ticker state if the user selected a ticker via button,
        # as the bot is no longer awaiting a typed response.
        if chat_id in awaiting_ticker:
            awaiting_ticker.remove(chat_id)
            
        await send_message(
            f"⏳ Fetching {ticker} news...",
            chat_id_override=chat_id, home_only=True)
        asyncio.create_task(fetch_and_send_news(chat_id, ticker))

    elif data.startswith("followup_"):
        ticker = data.split("_")[1]
        prev_summary = callback_query["message"]["text"]
        awaiting_follow_up[chat_id] = (ticker, prev_summary)
        await send_message(
            f"What would you like to follow up on for {ticker.upper()}?",
            chat_id_override=chat_id,
            home_only=True
        )

    elif data == "home":
        await send_inline_keyboard(chat_id)

async def fetch_and_send_news(chat_id: int, ticker: str):
    log(f"📰 Fetching news for {ticker} to chat {chat_id}")
    news_items = await fetch_latest_news(ticker)

    for item in news_items:
        summary = item["summary"]
        url = item.get("url")

        # FIX: Ensure all requested buttons (Account Summary, Fetch News, Read Article, Follow Up, Home) are present.
        keyboard = {
            "inline_keyboard": [
                # General Actions (The ones the user reported missing after news fetch)
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
            chat_id_override=chat_id,
            news_prompt=False,
            inline_keyboard=keyboard
        )

async def handle_message(chat_id: int, text: str):
    """Handle text messages."""
    log(f"📥 Received message from {chat_id}: {text[:30]}")
    
    if chat_id in awaiting_ticker:
        # Only fetch news if we are in the “awaiting ticker” state
        ticker = text.strip().upper()
        awaiting_ticker.remove(chat_id)
        await send_message(f"⏳ Fetching {ticker} news...", chat_id_override=chat_id, home_only=True)
        asyncio.create_task(fetch_and_send_news(chat_id, ticker))

    elif chat_id in awaiting_follow_up:
        # Only generate follow-up if we are in the “awaiting follow-up” state
        ticker, prev_summary = awaiting_follow_up.pop(chat_id)
        user_query = text.strip()
        
        response = await generate_follow_up_response(prev_summary, ticker, user_query, is_follow_up=True)
        
        await send_message(response, parse_mode="Markdown", chat_id_override=chat_id)

    elif text.lower().startswith("/start"):
        await send_inline_keyboard(chat_id)

async def process_update(update: dict):
    """Handle a single Telegram update."""
    if "message" in update:
        msg = update["message"]
        chat_id = msg["chat"]["id"]
        text = msg.get("text", "")
        await handle_message(chat_id, text)
    elif "callback_query" in update:
        asyncio.create_task(handle_callback(update["callback_query"]))

async def start_bot() -> None:
    """Main polling loop."""
    offset = 0
    log("🤖 Starting Telegram bot polling")
    while True:
        try:
            resp = await http_client.get(f"{BASE_URL}/getUpdates", params={"offset": offset + 1, "timeout": 20})
            if resp.status_code != 200:
                log(f"⚠️ Telegram polling error {resp.status_code}")
                await asyncio.sleep(3)
                continue
            updates = resp.json().get("result", [])
            if not updates:
                continue
            offset = updates[-1]["update_id"]
            await asyncio.gather(*(process_update(u) for u in updates))
        except httpx.RequestError:
            log("⚠️ Network error in bot polling")
            await asyncio.sleep(5)

async def close_http_client():
    log("🛑 Closing HTTP client")
    await http_client.aclose()
