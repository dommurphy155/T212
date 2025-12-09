import asyncio
import time
import re
import os
import pytz
import requests
import yfinance as yf 
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
import feedparser
from playwright.async_api import async_playwright, Browser, Page
from sklearn.feature_extraction.text import TfidfVectorizer
from contextlib import asynccontextmanager
from portfolio import fetch_all_data, build_summary
from dotenv import load_dotenv

load_dotenv()

try:
    import trafilatura
except ImportError:
    
    raise ImportError("Install dependencies: pip install trafilatura beautifulsoup4 requests python-dotenv openai playwright yfinance")

import openai

# === Config ===
GOOGLE_NEWS_BASE_URL = "https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en"
MAX_HEADLINES = 1
HOURS_LOOKBACK = 4
MAX_CONCURRENT = 3
TIMEOUT_MS = 6000
url_cache: Dict[str, float] = {}  # {url: timestamp of addition}
CACHE_TTL = 24 * 3600  # 24 hours in seconds

OPENAI_KEYS = [os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_API_KEY_BK")]
# Removed reliance on ALPHA_VANTAGE_API_KEY
# ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

browser_instance: Optional[Browser] = None
browser_launch_time: Optional[float] = None
BROWSER_REFRESH_INTERVAL = 2 * 3600  # 2 hours

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


@asynccontextmanager
async def get_browser():
    """
    Persistent global browser context manager.
    Reuses the same Chromium instance for all operations.
    Automatically refreshes after BROWSER_REFRESH_INTERVAL seconds.
    """
    global browser_instance, browser_launch_time

    now = time.time()
    if not browser_instance or not browser_launch_time or (now - browser_launch_time) > BROWSER_REFRESH_INTERVAL:
        # Launch fresh browser
        if browser_instance:
            try:
                await browser_instance.close()
                log("♻️ Old browser instance closed (refresh cycle)")
            except Exception:
                pass

        log("🚀 Launching persistent Playwright browser")
        p = await async_playwright().start()
        # Use a stable browser launch configuration
        browser_instance = await p.chromium.launch(headless=True)
        browser_launch_time = now

    try:
        yield browser_instance
    finally:
        # do not close here — persistent
        pass


# === Seen/Failed URL Cache ===
def add_to_cache(url: str):
    """Add URL to cache with current timestamp."""
    url_cache[url] = time.time()

def is_cached(url: str) -> bool:
    """Check if URL is in cache and still valid."""
    ts = url_cache.get(url)
    if ts and time.time() - ts < CACHE_TTL:
        return True
    elif ts:
        del url_cache[url]  # expired, remove
    return False

async def cleanup_cache_periodically():
    """Background task to clean expired URLs every hour."""
    while True:
        now = time.time()
        to_delete = [url for url, ts in url_cache.items() if now - ts >= CACHE_TTL]
        for url in to_delete:
            del url_cache[url]
        await asyncio.sleep(3600)  # run every hour

# === Utilities ===
def clean_text(text: str) -> str:
    log("🧹 Cleaning article text")
    text = re.sub(r"(?i)(terms of service|privacy policy|subscribe|cookies|disclaimer).*", "", text)
    text = re.sub(r"© ?\d{4}.*", "", text)
    text = re.sub(r"[•·►▶■◆●▪▫]", "", text)
    text = re.sub(r"\b(View More|Read more|Advertisement|Sign in|Watch Live|Skip to)\b.*", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([.!?])\s*", r"\1 ", text)  # normalize punctuation spacing
    text = text.strip()
    return text

def decode_google_redirect(url: str) -> str:
    log(f"🔗 Decoding Google News URL: {url[:30]}…")
    import urllib.parse
    if "news.google.com/rss/articles/" not in url:
        return url
    parsed = urllib.parse.urlparse(url)
    q = urllib.parse.parse_qs(parsed.query)
    target = next((v for k, v in q.items() if k in ["url", "q"]), None)
    return target[0] if target else None

def extract_page_text(url: str) -> str:
    log(f"📄 Extracting text from {url[:30]}…")
    try:
        text = None
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if text:
                text = clean_text(text)

        if not text or len(text.split()) < 50:
            html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10).text
            soup = BeautifulSoup(html, "html.parser")
            for t in soup(["script", "style", "noscript", "header", "footer", "form", "nav", "aside"]):
                t.decompose()
            paragraphs = [p.get_text() for p in soup.find_all("p") if len(p.get_text().split()) > 30]
            text = clean_text(" ".join(paragraphs))

        # Skip junk content
        if any(x in text.lower() for x in ["cookie", "privacy policy", "consent", "gdpr"]):
            log("🚫 Skipping article with consent/privacy content")
            return ""
        return text
    except Exception as e:
        log(f"❌ Failed to extract text: {e}")
        return ""

def summarize_text(text: str, top_n: int = 8) -> str:
    log("📝 Summarizing article content")
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s for s in sentences if 40 < len(s) < 500 and not any(x in s.lower() for x in ["cookie", "policy", "login", "subscribe"])]
    if not sentences:
        return ""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(sentences)
    scores = tfidf.sum(axis=1).A1
    best_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    best_sentences = [sentences[i] for i in best_idx]
    return " ".join(best_sentences).strip()

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

# === Playwright URL Resolution ===
async def resolve_google_news_article(browser: Browser, url: str) -> str:
    log(f"🌐 Resolving Google News URL: {url[:30]}…")
    page: Page = await browser.new_page()
    final_url = None
    try:
        await page.goto(url, timeout=TIMEOUT_MS)
        for _ in range(3):
            if "consent.google.com" in page.url:
                try:
                    # Attempt to click accept button if consent page is shown
                    await page.click('button[aria-label="Accept all"]')
                    await page.wait_for_load_state("networkidle")
                except:
                    pass
                await page.wait_for_timeout(120)
                await page.reload()
            else:
                break
        await page.wait_for_timeout(120)
        final_url = page.url

        blocked_domains = [
            "https://chat.whatsapp.com",
            "consent.yahoo.com",
            "consent.google.com",
            "privacyportal.onetrust.com",
            "cookie", "gdpr", "login", "auth"
        ]
        if any(x in final_url for x in blocked_domains):
            log(f"🚫 Skipping blocked URL: {final_url[:30]}…")
            return None

        if final_url and "news.google.com/rss/articles/" in final_url:
            return None

    except Exception:
        final_url = None
    finally:
        await page.close()
    return final_url

# === Fetch ticker headlines ===
async def fetch_ticker_headlines(ticker: str, exclude_urls: List[str] = []) -> List[Dict]:
    log(f"📰 Fetching headlines for {ticker}")
    url = GOOGLE_NEWS_BASE_URL.format(query=ticker)
    feed = feedparser.parse(url)
    now = datetime.utcnow()
    cutoff = now - timedelta(hours=HOURS_LOOKBACK)
    headlines = []
    for entry in feed.entries:
        if len(headlines) >= MAX_HEADLINES:
            break
        if entry.link in exclude_urls:
            continue
        published_parsed = entry.get("published_parsed")
        if not published_parsed:
            continue
        published_dt = datetime.fromtimestamp(time.mktime(published_parsed))
        if published_dt <= cutoff:
            continue
        headlines.append({"title": entry.title, "url": entry.link, "published": published_dt})
    return headlines

# === AI Summary with memory ===
async def generate_ai_summary(combined_summary: str, ticker: str) -> str:
    log(f"🤖 Generating AI summary for {ticker}")
    if not combined_summary.strip():
        return f"⚠️ No recent content found for {ticker.upper()}."

    # === Live price and portfolio context ===
    live_price_note = fetch_live_price(ticker) # Now returns a formatted string

    # === Use cached portfolio instead of live fetch ===
    try:
        data_out, net_deposits, missing = await fetch_all_data()

        # Build the comprehensive summary text using the new format
        summary_text = await build_summary(data_out, net_deposits, missing)
        
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
        log(f"Error fetching portfolio for news summary: {e}")
        holdings_msg = "(⚠️ Portfolio data unavailable)"
        portfolio_context = f"No portfolio cache. Assume user may hold {ticker.upper()}."

    # === Include historical AI memory context ===
    past_context = "\n\n".join(ai_memory[ticker][-10:]) if ai_memory[ticker] else "No prior summaries available."
    
    # === Prompt Template (Modified slightly for clarity) ===
    prompt = f"""
You are an experienced investment analyst whose job is to educate and inform, not to sell or hype. Your communication style is direct, honest, and grounded in facts. You explain complex concepts clearly and admit uncertainty when appropriate.

Your goal: Help the user understand what's happening, why it matters, and how to think about it—not just tell them what to do. Make sure to reference previous updates when relevant and maintain continuity.

=== CONTEXT ===

{live_price_note}

📊 Relevant Portfolio Context:
{portfolio_context}

📚 Previous {ticker.upper()} Updates (for continuity):
{past_context if ai_memory[ticker] else "No prior context—this is the first update for this asset."}

📰 Today's News Summary:
{combined_summary}

=== YOUR TASK ===

Analyze this news and create a Telegram message (max 275 words) that:

1. **Explains what actually happened** (the news event itself)
2. **Why it matters** (impact on the company, industry, or market)
3. **How it connects to the user's holdings** (direct or indirect effects)
4. **What to watch next** (follow-up events, triggers, risks, or opportunities)

=== CRITICAL RULES ===
... (Rules remain unchanged)

=== MESSAGE STRUCTURE ===

Use Telegram markdown. Keep it tight and readable:
🗞 **[Headline in 8-10 words] [Rate the news as positive 🔥, neutral ⚖️, or negative 🚨 for the company/your portfolio. One word]**


🔎 *What Happened* [2-3 sentences: the actual news event]

💡 *Why It Matters* [2-3 sentences: impact on company, industry, or your holdings. Define any jargon inline.]

📊 *Your Portfolio Context* [How this affects their positions together. Reference holdings specifically. If they don't own it, mention indirect effects.]

🔮 *What to Watch Next* [Specific dates, triggers, competitor moves, actionable signals, where to watch, how to avoid the hype and get the real stuff]

💭 *Final Take* - 🔹 **Key News Point:** [One bullet — most important thing to watch and where to watch next from today's news]  
- 🔹 **Portfolio Highlight:** [One bullet — most critical insight from portfolio context]  
- 🔹 **Overall Portfolio Rating:** [Concise health check, e.g., "You're looking healthy at the moment" or "Portfolio is majorly unbalanced; consider rebalancing X to Z". give good advice based on portfolio]
... (Tone Guidelines remain unchanged)
...
"""
    # === OpenAI Key Fallback System ===
    for i, key in enumerate(OPENAI_KEYS):
        if not key:
            continue
        key_label = "Primary" if i == 0 else f"Backup {i}"
        try:
            log(f"🔑 Using {key_label} OpenAI key for {ticker}")
            openai.api_key = key
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=950,
            )

            summary = response.choices[0].message.content.strip()

            # Maintain AI memory (last 100 entries)
            ai_memory[ticker].append(summary)
            ai_memory[ticker] = ai_memory[ticker][-100:]

            log(f"✅ {key_label} OpenAI key succeeded for {ticker}")
            return summary

        except Exception as e:
            log(f"❌ {key_label} OpenAI key failed: {e}")
            continue

    log(f"🚫 All OpenAI keys failed for {ticker}")
    return "🚫 All OpenAI keys failed — no summary generated."

# === Main entry for Telegram ===
# === Updated fetch_latest_news with cache integration (Logic remains the same) ===
async def fetch_latest_news(ticker: str) -> List[Dict]:
    """
    Returns a list of dicts: [{'summary': ai_summary, 'url': article_url}]
    """
    log(f"🔍 Fetching news for {ticker}")
    
    # Skip ambiguous queries
    ambiguous_queries = ["what", "why", "how", "who"]
    if ticker.lower() in ambiguous_queries:
        log(f"⚠️ Ambiguous query '{ticker}' skipped for news fetch")
        return [{"summary": f"⚠️ '{ticker}' is too vague. Please provide a specific ticker or company.", "url": None}]

    successful = []

    async with get_browser() as browser:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        async def sem_resolve(url):
            async with semaphore:
                decoded = decode_google_redirect(url)
                return decoded or await resolve_google_news_article(browser, url)

        while len(successful) < MAX_HEADLINES:
            remaining = MAX_HEADLINES - len(successful)
            new_headlines = await fetch_ticker_headlines(
                ticker, exclude_urls=[u for u in url_cache.keys()]
            )
            new_headlines = [h for h in new_headlines if not is_cached(h["url"])]

            if not new_headlines:
                break

            to_process = new_headlines[:remaining]
            tasks = [sem_resolve(h["url"]) for h in to_process]
            final_urls = await asyncio.gather(*tasks)

            for h, final_url in zip(to_process, final_urls):
                add_to_cache(h["url"])

                if not final_url:
                    continue

                text = extract_page_text(final_url)
                if not text:
                    continue

                summary = summarize_text(text)
                if summary:
                    successful.append({"summary": summary, "url": final_url})

            if len(successful) >= MAX_HEADLINES:
                break
            await asyncio.sleep(1)

    if not successful:
        return [{"summary": "🚫 No valid news summaries found. Try again later.", "url": None}]

    results = []
    for item in successful:
        ai_summary = await generate_ai_summary(item["summary"], ticker)
        results.append({"summary": ai_summary, "url": item["url"]})

    return results
