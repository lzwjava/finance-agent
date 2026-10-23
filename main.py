#!/usr/bin/env python3
"""FinanceIQ CLI - chat with the LlamaIndex finance agent.

Usage:
  python3 main.py                      # interactive chat
  python3 main.py "What should I invest in today?"
  python3 main.py --no-verbose "ideas triggered by today's AI earnings news"
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()  # optional .env support

from finance_agent.agent import build_agent
from finance_agent.llm import get_llm
from llama_index.core.base.llms.types import ChatMessage


def _parse():
    p = argparse.ArgumentParser(description="FinanceIQ - news-driven investment ideas (LlamaIndex agent)")
    p.add_argument("prompt", nargs="?", default=None, help="One-shot question; omit for interactive chat")
    p.add_argument("--no-verbose", action="store_true", help="Hide agent step logs")
    return p.parse_args()


async def ask_once(agent, user_msg: str, chat_history: list[ChatMessage] | None = None) -> str:
    handler = agent.run(user_msg=user_msg, chat_history=chat_history or [])
    result = await handler
    return str(result)


async def interactive(agent) -> None:
    print("=" * 64)
    print(" FinanceIQ - LlamaIndex finance agent (DeepSeek + live web news)")
    print(" Ask what to invest in, or follow up on an idea. Ctrl-D to exit.")
    print("=" * 64)
    history: list[ChatMessage] = []
    while True:
        try:
            user_msg = input("\n\U0001f4c8 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return
        if not user_msg:
            continue
        if user_msg.lower() in {"exit", "quit"}:
            print("Bye!")
            return
        print("... reasoning (searching news, checking quotes) ...")
        try:
            reply = await ask_once(agent, user_msg, history)
        except Exception as exc:
            print(f"\n[error] {exc}")
            continue
        history.append(ChatMessage(role="user", content=user_msg))
        history.append(ChatMessage(role="assistant", content=str(reply)))
        print(f"\n\U0001f4ca FinanceIQ:\n{reply}")


async def main() -> int:
    args = _parse()
    try:
        llm = get_llm()
    except RuntimeError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 1
    agent = build_agent(llm=llm, verbose=not args.no_verbose)

    if args.prompt:
        print("... reasoning (searching news, checking quotes) ...")
        reply = await ask_once(agent, args.prompt)
        print(f"\n{reply}")
    else:
        await interactive(agent)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))