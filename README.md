# FinanceIQ — news-driven investment research agent

A [LlamaIndex](https://llamaindex.ai) agent that reads today's financial news
(Yahoo Finance, Reuters, CNBC, Bloomberg, ...) and turns it into grounded
investment ideas with tickers, theses, risks and position sizing.

```
You → FinanceIQ → search_market_news() × N   (live web news via Tavily)
                 → get_stock_quote(ticker)   (live prices via yfinance)
                 → synthesizes structured ideas
```

## Architecture

| Module | Role |
|---|---|
| `finance_agent/agent.py` | `FunctionAgent` (LlamaIndex agent framework) + system prompt |
| `finance_agent/tools.py` | Agent tools: Tavily news search, yfinance quotes/indices |
| `finance_agent/llm.py` | `DeepSeekLLM` — OpenAI-compatible LLM adapter for llama-index |
| `main.py` | CLI (one-shot or interactive chat) |

**Stack:** LlamaIndex `FunctionAgent` (tool-calling agent), DeepSeek (`deepseek-chat`,
OpenAI-compatible API), Tavily (`topic=news` search API), yfinance.

> **Why Tavily instead of a built-in "LLM web search"?** The provider web-search
> tools were **not usable in this environment**: the `OPENAI_API_KEY` is rejected by
> api.openai.com, `ANTHROPIC_API_KEY` is empty (and the Claude relay blocks
> third-party clients), and the Gemini key has depleted credits. DeepSeek/Groq
> don't expose native web search. Tavily is the standard agentic web-search API
> and was verified live here.

## Setup

```bash
# 1. Install deps (system Python, your choice)
pip install --break-system-packages -r requirements.txt   # macOS/PEP 668 only

# 2. Keys (already exported in ~/.zshrc on this machine, or use a .env)
cp .env.example .env   # fill DEEPSEEK_API_KEY + TAVILY_API_KEY
#    .env is auto-loaded if present
```

## Usage

```bash
# One-shot question
python3 main.py "What should I invest in today?"

# Interactive chat (keeps conversation context across turns)
python3 main.py

# Hide agent step traces
python3 main.py --no-verbose "..."
```

Example questions:

- "What are the strongest sectors in the news right now, and what's a direct
  way to express that positioning?"
- "Is there a cheaper way to play the AI theme than the obvious large caps?"
- "What should I avoid buying right now based on today's headlines?"

## How the agent works

1. **Grounded in news** — the system prompt forces 2-3 `search_market_news`
   calls (macro, sectors, plus `site:finance.yahoo.com ...`) before answering.
2. **Verified prices** — every ticker is checked with `get_stock_quote` before
   being recommended.
3. **Structured output** — market snapshot → 3-5 ideas (thesis / catalysts /
   risks / sizing / horizon) → "watch out for" list → education disclaimer.

## Caveats

- **No financial advice.** Output is educational; always verify before investing.
- Search results can lag or miss articles; the agent cites what it actually retrieved.
- `yfinance` `fast_info` may return the **opening** price instead of a live trade
  outside market hours — the agent is instructed to flag this and you should
  cross-check levels before acting.
- API keys must have credit; a dead key surfaces as a clear config error at startup.

## Files

```
finance_agent/
  __init__.py
  agent.py        # FunctionAgent + system prompt + tool wiring
  tools.py        # search_market_news, get_company_news, get_stock_quote, get_market_indices
  llm.py          # DeepSeekLLM (OpenAI-compatible, llama-index)
main.py           # CLI
requirements.txt
.env.example
```