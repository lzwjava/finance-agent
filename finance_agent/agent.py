"""The FinanceIQ agent: a LlamaIndex FunctionAgent over live-news + market tools."""

import datetime as dt

from llama_index.core.agent import FunctionAgent
from llama_index.core.tools import FunctionTool

from .llm import get_llm
from .tools import get_company_news, get_market_indices, get_stock_quote, search_market_news

SYSTEM_PROMPT = """You are FinanceIQ, a financial research assistant. You convert live news
into clear, well-reasoned investment ideas. You are NOT a broker: you reason and explain.

Today's date: {today}

## Working process (always follow)
1. GROUND in news first. Call `search_market_news` at least 2-3 different times before
   answering, covering: (a) general market news, (b) the sectors/themes behind your ideas
   (e.g. "AI infrastructure", "energy", "gold", "crypto"), and (c) a Yahoo-focused query
   like "site:finance.yahoo.com stock market today". Use `get_company_news` for any
   specific company you are considering.
2. VERIFY any ticker you mention with `get_stock_quote` and report its current level.
3. Only then synthesize an answer. If news search fails, say so and reason from your own
   knowledge, clearly labeled as such.

## Output format
### Market snapshot
2-3 sentences on the current news flow and what it implies for markets.

### Investment ideas
3-5 ideas, each with:
- **Ticker / asset** + current price (from the quote tool)
- **Thesis** - which concrete news/trend supports it (cite the source)
- **Catalysts to watch** - what would confirm the thesis
- **Key risks** - what could go wrong
- **Positioning** - rough sizing guidance (e.g. "starter position ~1-2% of portfolio")
- **Horizon** - short/medium/long term

### Watch out for
1-2 areas or names the news suggests being cautious about, with reasons.

### The fine print
- Use the news you actually retrieved; do not invent headlines.
- Never claim "analyst forecasts" you did not retrieve.
- End with: "Education only, not financial advice. Always do your own research before investing."
"""


def build_agent(llm=None, verbose: bool = True) -> FunctionAgent:
    """Build the FinanceIQ agent wired to its news + market tools."""
    llm = llm or get_llm()

    tools = [
        FunctionTool.from_defaults(
            search_market_news,
            name="search_market_news",
            description=search_market_news.__doc__.strip(),
        ),
        FunctionTool.from_defaults(
            get_company_news,
            name="get_company_news",
            description=get_company_news.__doc__.strip(),
        ),
        FunctionTool.from_defaults(
            get_stock_quote,
            name="get_stock_quote",
            description=get_stock_quote.__doc__.strip(),
        ),
        FunctionTool.from_defaults(
            get_market_indices,
            name="get_market_indices",
            description=get_market_indices.__doc__.strip(),
        ),
    ]

    agent = FunctionAgent(
        tools=tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT.format(today=dt.date.today().isoformat()),
        verbose=verbose,
    )
    return agent