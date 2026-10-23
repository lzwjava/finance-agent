"""Agent tools: live news search (Tavily) and stock quotes (yfinance)."""

import datetime as dt
import os
from typing import Optional

import requests


# --------------------------------------------------------------------------- #
# News search (Tavily)
# --------------------------------------------------------------------------- #
def _tavily_search(
    query: str,
    topic: str = "news",
    max_results: int = 6,
    days: int = 3,
) -> list[dict]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set.")
    payload = {
        "api_key": api_key,
        "query": query,
        "topic": topic,
        "max_results": max_results,
        "days": days,
        "include_answer": False,
        "include_raw_content": False,
    }
    resp = requests.post("https://api.tavily.com/search", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])


def search_market_news(query: str, max_results: int = 6, days: int = 3) -> str:
    """Search the live web for recent financial news.

    Returns a numbered list of headlines with source, publication date, URL and
    a short summary. Use it to ground recommendations in what is actually
    happening (Yahoo Finance, Reuters, CNBC, Bloomberg, etc.). You can search
    for sectors ('AI infrastructure', 'renewable energy', 'banks'), themes
    ('Fed rate cut', 'chip export controls'), or single tickers
    ('NVIDIA stock news'). Prefer `news` topic results. Include
    'site:finance.yahoo.com' in the query to focus on Yahoo Finance articles.
    """
    try:
        results = _tavily_search(query, "news", max_results, days)
    except Exception as exc:  # surface a clean error to the LLM
        return f"Error searching news: {exc}"
    if not results:
        return "No recent news results found for this query. Try a different, broader query."
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "").strip()
        url = r.get("url", "")
        date = (r.get("published_date") or "")[:10]
        content = (r.get("content") or "").strip().replace("\n", " ")
        if len(content) > 400:
            content = content[:397] + "..."
        lines.append(f"{i}. {title}\n   source: {url}  ({date})\n   {content}")
    return "\n".join(lines)


def get_company_news(ticker: str, days: int = 7, max_results: int = 5) -> str:
    """Search the live web for recent news about one company / ticker."""
    return search_market_news(f"{ticker} stock news", max_results=max_results, days=days)


# --------------------------------------------------------------------------- #
# Market data (yfinance)
# --------------------------------------------------------------------------- #
def get_stock_quote(ticker: str) -> str:
    """Fetch a live quote for a ticker (stock, ETF, index or crypto).

    Returns current price, previous close, day change, and the 1-month
    high/low. Use this to check prices before recommending a ticker.
    """
    import yfinance as yf

    ticker = ticker.strip().upper()
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info

        price = fi.get("last_price")
        prev_close = fi.get("previous_close")
        open_price = fi.get("open")

        last_result = t.last_result if hasattr(t, "last_result") else {}
        currency = last_result.get("currency") or ""

        day_change = day_change_pct = None
        if price is not None and prev_close:
            day_change = price - prev_close
            day_change_pct = (day_change / prev_close) * 100.0

        hist = t.history(period="1mo")
        high, low = None, None
        if hist is not None and not hist.empty and "High" in hist and "Low" in hist:
            high = float(hist["High"].max())
            low = float(hist["Low"].min())

        fmt = lambda v, p=2: f"{v:,.{p}f}" if v is not None else "n/a"
        lines = [
            f"Quote for {ticker} ({currency})",
            f"- Last price: {fmt(price)}",
            f"- Open: {fmt(open_price)}",
            f"- Previous close: {fmt(prev_close)}",
            f"- Day change: {fmt(day_change)} ({fmt(day_change_pct)}%)",
            f"- 1-month range: {fmt(low)} - {fmt(high)}",
        ]
        return "\n".join(lines)
    except Exception as exc:
        return f"Error fetching quote for {ticker}: {exc}. Check the ticker symbol."


def get_market_indices() -> str:
    """Fetch live levels of the main US market indices (S&P 500, Nasdaq, Dow)."""
    out = []
    for ticker, name in [
        ("^GSPC", "S&P 500"),
        ("^IXIC", "Nasdaq Composite"),
        ("^DJI", "Dow Jones"),
    ]:
        try:
            quote = get_stock_quote(ticker)
            out.append(f"{name}:\n" + quote.replace(f"Quote for {ticker}", "  ") + "\n")
        except Exception as exc:
            out.append(f"{name}: error ({exc})")
    return "\n".join(out).strip()


def _today() -> str:
    return dt.date.today().isoformat()