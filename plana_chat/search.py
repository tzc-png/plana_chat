"""网络搜索（可选增强，默认关闭）。

当前默认用 DuckDuckGo HTML 结果页（走代理），best-effort 解析摘要。
如需更稳定可换成带 API key 的搜索服务（Tavily / SerpAPI / Bing 等）。
"""
from __future__ import annotations

import html
import re

import httpx

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)


async def duckduckgo_search(query: str, proxy: str = "", max_results: int = 5) -> str:
    """DuckDuckGo HTML 搜索，返回拼接的摘要文本（每行一条）。"""
    kwargs: dict = {"timeout": 20, "headers": {"User-Agent": _UA}, "trust_env": False}
    if proxy:
        kwargs["proxy"] = proxy

    async with httpx.AsyncClient(**kwargs) as client:
        resp = await client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
        )
        resp.raise_for_status()

    snippets = re.findall(r'result__snippet[^>]*>(.*?)</a>', resp.text, re.S)
    results: list[str] = []
    for s in snippets:
        s = re.sub(r"<[^>]+>", "", s)
        s = html.unescape(s).strip()
        if s and s not in results:
            results.append(s)
        if len(results) >= max_results:
            break

    return "\n".join(f"- {r}" for r in results) if results else ""


async def web_search(query: str, cfg) -> str:
    provider = cfg.search.get("provider", "duckduckgo")
    proxy = cfg.telegram.get("proxy", "")
    max_results = cfg.search.get("max_results", 5)
    if provider == "duckduckgo":
        return await duckduckgo_search(query, proxy=proxy, max_results=max_results)
    return ""
