"""FinanceIQ - a LlamaIndex agent that reads live financial news and suggests investments.

Uses:
  * LlamaIndex FunctionAgent (agent framework + tool calling)
  * DeepSeek (OpenAI-compatible API) as the reasoning LLM
  * Tavily web search for live news (Yahoo Finance / Reuters / CNBC / ...)
  * yfinance for real-time stock quotes
"""

__version__ = "0.1.0"