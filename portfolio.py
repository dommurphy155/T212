import os
import time
import json
import asyncio
import httpx
import pytz
import yfinance as yf
from typing import Any, Dict, List, Tuple, Optional
from dotenv import load_dotenv
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs

load_dotenv()

AUTH_HEADER = os.getenv("T212_AUTH_HEADER")
if not AUTH_HEADER:
    raise RuntimeError("❌ Missing T212_AUTH_HEADER in .env file")

BASE_URL = "https://live.trading212.com/api/v0/equity"
TRANSACTIONS_URL = "https://live.trading212.com/api/v0/history/transactions"

ENDPOINTS = [
    "account/cash",
    "portfolio",
    "metadata/instruments",
    "metadata/exchanges",
    "pies",
    "history/orders",
]

HEADERS = {"Authorization": AUTH_HEADER, "Accept": "application/json"}

CACHE_FILE = os.path.expanduser("~/portfolio_cache.json")

# === Global cache ===
_cached_data: Optional[Dict[str, Any]] = None
_cached_net_deposits: float = 0.0
_cached_missing: List[str] = []
_cache_timestamp: float = 0.0
_cache_refresh_lock = asyncio.Lock()
_cache_initialized = False


def log(msg: str, status: str = ""):
    uk_time = datetime.now(pytz.timezone("Europe/London"))
    ts = uk_time.strftime("%H:%M:%S")
    status_str = f"{status} " if status else ""
    print(f"[{ts}] {status_str}{msg}")

def normalize_ticker(ticker: str) -> str:
    if not ticker:
        return ticker
    # Removes the suffix for display purposes and cleaning for yfinance
    return ticker.replace(" ", "").replace("_US_EQ", "").replace("_EQ", "").replace(".", "").upper()

# === Cache file persistence ===
def save_cache_to_disk():
    global _cached_data, _cached_net_deposits, _cached_missing, _cache_timestamp
    try:
        payload = {
            "timestamp": _cache_timestamp,
            "net_deposits": _cached_net_deposits,
            "missing": _cached_missing,
            "data": _cached_data,
        }
        with open(CACHE_FILE, "w") as f:
            json.dump(payload, f)
        log(f"Cache saved to disk ({CACHE_FILE})", "💾")
    except Exception as e:
        log(f"Failed to save cache: {e}", "❌")


def load_cache_from_disk() -> bool:
    global _cached_data, _cached_net_deposits, _cached_missing, _cache_timestamp, _cache_initialized
    if not os.path.exists(CACHE_FILE):
        return False

    try:
        with open(CACHE_FILE, "r") as f:
            payload = json.load(f)

        if not isinstance(payload, dict) or "data" not in payload:
            log("Cache file invalid or corrupt — ignoring it", "⚠️")
            return False

        _cached_data = payload.get("data", {})
        _cached_net_deposits = payload.get("net_deposits", 0.0)
        _cached_missing = payload.get("missing", [])
        _cache_timestamp = payload.get("timestamp", 0.0)
        _cache_initialized = True

        age = time.time() - _cache_timestamp
        log(f"Loaded cache from disk (age: {int(age)}s)", "🧠")
        return True

    except Exception as e:
        log(f"Failed to load cache from disk: {e}", "❌")
        return False


async def fetch(client: httpx.AsyncClient, endpoint: str, max_retries: int = 5, backoff: float = 1.5) -> Optional[Dict[str, Any]]:
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.get(url, timeout=5)
            if resp.status_code == 200:
                return resp.json()
            else:
                log(f"{endpoint} failed: HTTP {resp.status_code}", "❌")
        except httpx.RequestError:
            log(f"{endpoint} network error, retrying...", "❌")
        await asyncio.sleep(min(backoff ** attempt, 5))
    log(f"{endpoint} failed after {max_retries} retries", "❌")
    return None


async def fetch_transactions(client: httpx.AsyncClient) -> float:
    """
    Fetches all transactions and aggregates net deposits (Deposits - Withdrawals).
    Uses a higher limit to reduce pagination requests.
    """
    log("Fetching transaction history...")
    total_net_deposits = 0.0
    # Increase limit to 50 to capture more data per request
    params: Dict[str, Any] = {"limit": 50} 
    page_count = 0

    while True:
        try:
            resp = await client.get(TRANSACTIONS_URL, headers=HEADERS, params=params, timeout=10)
            if resp.status_code != 200:
                log(f"Transaction fetch failed: HTTP {resp.status_code} on page {page_count+1}", "❌")
                break
            data = resp.json()
            if not isinstance(data, dict):
                log("Invalid transaction response format", "❌")
                break
            
            page_count += 1

        except httpx.RequestError:
            log("Transaction fetch network error", "❌")
            break

        items = data.get("items", [])
        if not items and page_count == 1:
            log("No transactions found.", "ℹ️")
            break

        for item in items:
            txn_type = item.get("type")
            amount = float(item.get("amount", 0))
            
            if txn_type == "DEPOSIT":
                total_net_deposits += amount
            elif txn_type == "WITHDRAWAL":
                total_net_deposits -= amount

        next_page = data.get("nextPagePath")
        if next_page:
            query_params = parse_qs(urlparse(next_page).query)
            cursor = query_params.get("cursor", [None])[0]
            
            if cursor:
                params = {"cursor": cursor, "limit": 50}
                # log(f"Page {page_count} done. Fetching next...", "📄")
            else:
                break
        else:
            log(f"Finished fetching transactions. Total pages: {page_count}", "✅")
            break

    return total_net_deposits


FX_URL = "https://api.frankfurter.app/latest?from=USD&to=GBP"


async def fetch_exchange_rate(client: httpx.AsyncClient) -> float:
    rate = 1.0
    try:
        resp = await client.get(FX_URL, timeout=5)
        data = resp.json()
        fetched_rate = data.get("rates", {}).get("GBP")
        if fetched_rate is not None:
            return float(fetched_rate)
        else:
            log(f"GBP rate not found in response: {data}", "❌")
    except httpx.RequestError as e:
        log(f"Exchange rate fetch network error: {e}", "❌")
    except Exception as e:
        log(f"Unexpected error fetching FX rate: {e}", "❌")
    return rate

async def fetch_live_price(ticker: str) -> Tuple[float, str]:
    """
    Fetch the live price of a stock using yfinance.
    Returns (price, currency_code). Price is 0.0 if failed.
    """
    raw_symbol = normalize_ticker(ticker)
    
    symbols_to_try = [raw_symbol]

    if raw_symbol == "VWRPL":
        symbols_to_try.append("VWRP.L")
        symbols_to_try.append("VWRL.L")

    if "_US_EQ" not in ticker.upper():
        if not raw_symbol.endswith((".L", ".AS", ".DE")):
             symbols_to_try.append(f"{raw_symbol}.L")
             
    symbols_to_try = list(dict.fromkeys(symbols_to_try))
    
    def sync_fetch(symbol_list):
        for clean_symbol in symbol_list:
            try:
                data = yf.Ticker(clean_symbol).info
                current_price = data.get("regularMarketPrice")
                currency = data.get("currency")
                
                if current_price is None or current_price == 0:
                    current_price = data.get("previousClose")
                    
                if current_price is not None and current_price > 0:
                    log(f"✅ Price found for {raw_symbol} using yf symbol {clean_symbol}")
                    return current_price, currency or ("GBP" if clean_symbol.endswith(".L") else "USD") 
                
            except Exception:
                pass

        log(f"⚠️ No price found for {raw_symbol} in yfinance data.", "⚠️")
        return 0.0, ""

    current_price_raw, currency = await asyncio.to_thread(sync_fetch, symbols_to_try)
    return current_price_raw, currency


async def _refresh_cache():
    global _cached_data, _cached_net_deposits, _cached_missing, _cache_timestamp, _cache_initialized

    log("Refreshing portfolio cache", "🔄")
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    async with httpx.AsyncClient(headers=HEADERS, http2=True, limits=limits) as client:
        endpoint_tasks = {ep: asyncio.create_task(fetch(client, ep)) for ep in ENDPOINTS}
        results_task = asyncio.gather(*endpoint_tasks.values())
        deposits_task = asyncio.create_task(fetch_transactions(client))
        fx_task = asyncio.create_task(fetch_exchange_rate(client))

        results, net_deposits, exchange_rate = await asyncio.gather(
            results_task, deposits_task, fx_task
        )

        data = dict(zip(ENDPOINTS, results))
        missing = [ep for ep, res in data.items() if res is None]
        data["exchange_rate"] = exchange_rate or 1.0

        _cached_data = data
        _cached_net_deposits = net_deposits
        _cached_missing = missing
        _cache_timestamp = time.time()
        _cache_initialized = True

        save_cache_to_disk()
        log(f"Successfully refreshed portfolio cache", "✅")


async def start_cache_refresh_loop():
    global _cache_refresh_lock, _cached_data, _cache_timestamp, _cache_initialized

    log("Starting portfolio cache refresh loop (every 65s)", "🔁")

    loaded = load_cache_from_disk()
    cache_age = (time.time() - _cache_timestamp) if _cache_timestamp else None

    if loaded and cache_age is not None and cache_age < 65:
        log(f"Using existing cache (age: {int(cache_age)}s)", "🧠")
    else:
        log("No valid or recent cache found — refreshing now", "🔄")
        async with _cache_refresh_lock:
            await _refresh_cache()

    while True:
        await asyncio.sleep(65)
        async with _cache_refresh_lock:
            try:
                await _refresh_cache()
            except Exception as e:
                log(f"Cache refresh failed: {e}", "❌")


async def fetch_all_data() -> Tuple[Dict[str, Any], float, List[str]]:
    global _cached_data, _cached_net_deposits, _cached_missing, _cache_initialized, _cache_refresh_lock

    if not _cache_initialized:
        async with _cache_refresh_lock:
            if not _cache_initialized:
                log("Initializing cache with first refresh", "⚠️")
                await _refresh_cache()

    log(f"Using cached data (age: {time.time() - _cache_timestamp:.1f}s)")
    return _cached_data.copy(), _cached_net_deposits, _cached_missing.copy()


async def build_summary(data, net_deposits, missing):
    log("Generating detailed portfolio summary")

    portfolio = data.get("portfolio", []) or []
    exchange_rate = data.get("exchange_rate", 1.0)
    
    # Initialize aggregated totals
    total_invested_gbp = 0.0 # Cost Basis
    total_market_value_gbp = 0.0
    holdings_msg = ""
    
    def format_return_str(amount, percent):
         sign = '+' if amount >= 0 else ''
         # User prefers (2.0%) format over (+2.0%), so slightly adjusted
         return f"{sign}£{amount:,.2f} ({sign}{percent:.2f}%)"

    if portfolio:
        tasks = {h['ticker']: asyncio.create_task(fetch_live_price(h['ticker'])) for h in portfolio}
        live_prices_results = {t: await task for t, task in tasks.items()}

        for holding in portfolio:
            raw_ticker = holding.get("ticker", "Unknown")
            ticker = normalize_ticker(raw_ticker)
            shares = holding.get("quantity", 0.0)
            avg_price_native = holding.get("averagePrice", 0.0)
            
            fetched_price_raw, fetched_currency = live_prices_results.get(raw_ticker, (0.0, "GBP"))
            current_price_native = fetched_price_raw if fetched_price_raw > 0 else avg_price_native
            
            is_us_stock = "_US_EQ" in raw_ticker 
            
            # --- Calculation Block ---
            invested_native = avg_price_native * shares
            
            if is_us_stock:
                current_price_usd = current_price_native
                current_value_gbp = (shares * current_price_usd) * exchange_rate
                invested_value_gbp = invested_native * exchange_rate
                
                current_price_str = f"${current_price_usd:,.2f} (≈£{(current_price_usd * exchange_rate):,.2f})"
                avg_price_str = f"${avg_price_native:,.2f} (≈£{(avg_price_native * exchange_rate):,.2f})"
            else:
                current_price_gbp = current_price_native
                current_value_gbp = shares * current_price_gbp
                invested_value_gbp = invested_native
                
                current_price_str = f"£{current_price_gbp:,.2f}"
                avg_price_str = f"£{avg_price_native:,.2f}"

            ppl_gbp = current_value_gbp - invested_value_gbp
            pct_change = (ppl_gbp / invested_value_gbp * 100) if invested_value_gbp else 0.0
            
            total_market_value_gbp += current_value_gbp
            total_invested_gbp += invested_value_gbp
            
            holding_return_str = format_return_str(ppl_gbp, pct_change)
            
            holdings_msg += (
                f"• {ticker}\n"
                f"VALUE: £{current_value_gbp:,.2f}\n"
                f"INVESTED: £{invested_value_gbp:,.2f}\n"
                f"RETURN: {holding_return_str}\n"
                f"SHARES: {shares:.8f}\n"
                f"AVERAGE PRICE: {avg_price_str}\n"
                f"CURRENT PRICE: {current_price_str}\n\n"
            )

    if not holdings_msg:
        holdings_msg = "   • *No active holdings found.*\n"

    # --- CRITICAL FIX: Use 'free' cash instead of 'total' ---
    # 'total' in T212 API usually means Total Account Value (Cash + Invested) or Total Cash (Blocked + Free).
    # Based on your data, 'free' is the correct field for "Available Cash".
    available_cash = data.get("account/cash", {}).get("free", 0.0)

    # Calculate Total Portfolio Value (Holdings + Available Cash)
    total_portfolio_value = total_market_value_gbp + available_cash
    
    # Holdings Performance (Cost Basis vs Current Value)
    holdings_return_gbp = total_market_value_gbp - total_invested_gbp
    holdings_return_pct = (holdings_return_gbp / total_invested_gbp * 100) if total_invested_gbp else 0.0
    holdings_return_str = format_return_str(holdings_return_gbp, holdings_return_pct)
    
    # Overall Performance (Total Value vs Net Deposits)
    overall_return_gbp = total_portfolio_value - net_deposits
    overall_return_pct = (overall_return_gbp / net_deposits * 100) if net_deposits else 0.0
    overall_return_str = format_return_str(overall_return_gbp, overall_return_pct)


    # Header matching your requested format
    header_msg = (
        f"📊 Trading212 Portfolio Summary\n\n"
        f"Total Portfolio Value: £{total_portfolio_value:,.2f}\n" 
        f"Total Net Deposits: £{net_deposits:,.2f}\n"             
        f"Total Performance: {overall_return_str}\n" 
        f"\nHOLDINGS;\n"
    )
    
    # Footer
    footer_msg = (
        f"⸻\n"
        f"💷 Available Cash: £{available_cash:,.2f}\n"
        f"📈 Total Holdings Performance: {holdings_return_str}\n"
    )
    
    msg = header_msg + holdings_msg.strip() + "\n" + footer_msg

    if missing:
        msg += f"\n⚠️ *Missing data from T212 API*: {', '.join(missing)}\n"

    return msg
