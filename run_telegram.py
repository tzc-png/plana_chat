"""Telegram 机器人入口：python run_telegram.py"""
import asyncio

from plana_chat.telegram import main

if __name__ == "__main__":
    asyncio.run(main())
