import asyncio
import time
import os
import pytz
import requests
import yfinance as yf 
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from openai import AsyncOpenAI
from collections import defaultdict
from portfolio import fetch_all_data, build_summary

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("❌ No OPENROUTER_API_KEY in .env")

client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
# === Memory Store (per ticker) ===
ai_memory = defaultdict(list)

# === Logging ===
_log_count = 0

def log(msg: str):
    global _log_count
    _log_count += 1
    uk_time = datetime.now(pytz.timezone("Europe/London"))
    ts = uk_time.strftime("%H:%M:%S")
    print(f"[{ts}] [{_log_count}] {msg}")

# --- REFACTORED: Use Yfinance for Live Price Fetch ---
def fetch_live_price(ticker: str) -> str:
    log(f"💹 Fetching live price for {ticker}")
    clean_symbol = ticker.replace("_US_EQ", "").replace("_EQ", "")
    try:
        data = yf.Ticker(clean_symbol).info
        
        # Use regular market price
        price = data.get("regularMarketPrice")
        currency = data.get("currency")
        
        if price is None or price == 0:
            price = data.get("previousClose") # Fallback
            
        if price is None:
            log(f"⚠️ No price data for {ticker}")
            return f"⚠️ No price data available for {ticker}"
        
        log(f"💰 {ticker} price: {currency} {price:.2f}")
        return f"💵 Current {ticker.upper()} Price: {currency} {price:,.2f}"
    except Exception as e:
        log(f"❌ Price fetch failed for {ticker} using yfinance: {e}")
        return f"❌ Price fetch failed for {ticker}: {str(e)}"

async def generate_follow_up_response(prev_summary: str, ticker: str, user_query: str, is_follow_up: bool = False) -> str:
    log(f"🤖 Generating follow-up response for {ticker}")
    
    # Ensure this is a follow-up query (though checked by telebot state)
    if not is_follow_up:
        log(f"⚠️ Not a follow-up query for {ticker}: {user_query}")
        # Return a simple error message instead of the original code's misleading one
        return "⚠️ Internal error: Query not recognized as a follow-up. Please try again using the 'Follow Up' button."

    if not prev_summary.strip():
        return f"⚠️ No previous summary available for {ticker.upper()}."

    # === Live price and portfolio context ===
    live_price_note = fetch_live_price(ticker)
    
    try:
        # fetch_all_data returns 3 values (data, net_deposits, missing)
        portfolio_data, net_deposits, missing = await fetch_all_data() 
        # build_summary is called with 3 positional arguments
        summary_text = await build_summary(portfolio_data, net_deposits, missing) 
        
        # Extract only the relevant holdings section for the AI context
        holdings_start = summary_text.find("Each Ticker (full info):")
        holdings_end = summary_text.find("💷 Available Cash:")
        
        holdings_msg = (
            summary_text[holdings_start:holdings_end].strip()
            if holdings_start != -1 and holdings_end != -1
            else summary_text # Use full summary if bounds are tricky
        )
        
        user_ticker_line = next(
            (line.strip() for line in holdings_msg.splitlines() if f"• {ticker.upper()}" in line),
            None
        )
        
        portfolio_context = (
            f"You hold this stock. Tie the analysis directly to your position:\n{user_ticker_line}"
            if user_ticker_line else
            f"You do not currently hold {ticker.upper()}. Discuss how this news could present an opportunity or affect your other holdings indirectly."
        )

    except Exception as e:
        log(f"Error fetching portfolio for follow-up: {e}")
        holdings_msg = "(⚠️ Portfolio data unavailable)"
        portfolio_context = f"No live portfolio data. Assume user may hold {ticker.upper()}."

    # === Include historical AI memory context ===
    past_context = "\n\n".join(ai_memory[ticker][-10:]) if ai_memory[ticker] else "No prior summaries available."
    
    # === Prompt Template (Kept similar to original but using new context structure) ===
    prompt = f"""
You are a sharp, grounded investment analyst. Write a follow-up explanation to help the user understand the situation around {ticker.upper()} — what’s happening, why it matters, and what they should be aware of — based on the user’s follow-up message and the prior summary.  
Reply directly to the user’s message as if this is the last time you will ever speak to me. you must clear up everything straight away this is the only reply you get make the most out of it and cover everything — clean, confident, and structured using **Telegram markdown**.

=== CONTEXT ===
{live_price_note}

📊 Relevant Portfolio Context:
{portfolio_context}

📰 Previous News Summary:
{prev_summary}

🗣️ User Follow-up Message:
"{user_query}"

📚 Past {ticker.upper()} AI Memory:
{past_context if ai_memory[ticker] else "No past context available."}

=== TASK ===
- Address what the user asked or expressed confusion about first.  
- Clarify *why* it matters in the bigger picture (company, sector, macro).  
- If relevant, link it briefly to the user’s holdings.  
- Include additional context only if it helps understanding.  
- Admit uncertainty if data isn’t clear.  
- No fake confidence, no filler.
- Do not reproduce previous summaries verbatim.
"""
    try:
        log(f"🔑 Using OpenRouter API for {ticker}")
        # log(f"🧠 FOLLOW-UP INPUT PROMPT for {ticker}:\n{'='*60}\n{prompt}\n{'='*60}") # Debugging logging removed

        response = await client.chat.completions.create(
            # Using deepseek-r1t-chimera as it was already in the original code, but confirmed to be working.
            model="tngtech/deepseek-r1t-chimera:free", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=950,
        )

        summary = response.choices[0].message.content.strip()

        # log(f"🗣️ FOLLOW-UP OUTPUT RESPONSE for {ticker}:\n{'='*60}\n{summary}\n{'='*60}") # Debugging logging removed

        # Maintain AI memory (last 100 entries)
        ai_memory[ticker].append(summary)
        ai_memory[ticker] = ai_memory[ticker][-100:]

        log(f"✅ OpenRouter succeeded for {ticker}")
        return summary

    except Exception as e:
        log(f"❌ OpenRouter failed: {e}")
        return "🚫 OpenRouter failed — no response generated."
