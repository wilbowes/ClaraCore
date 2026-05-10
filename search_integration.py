"""
Web search and page fetch integration for Clara.
Uses Brave Search API for search, httpx + BeautifulSoup for page content.

Requires BRAVE_API_KEY environment variable.
Free tier: 2000 queries/month — https://api.search.brave.com/
"""

import os
import httpx
from bs4 import BeautifulSoup

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Tags to strip when extracting page text — navigation, ads, boilerplate
STRIP_TAGS = [
    "script", "style", "nav", "header", "footer", "aside",
    "form", "button", "iframe", "noscript", "svg",
]

# Max characters to return from a fetched page — keeps context cost reasonable
PAGE_CHAR_LIMIT = 4000


async def brave_search(query: str, count: int = 5) -> str:
    """
    Search the web via Brave Search API.
    Returns a formatted string of results suitable for Clara's context.
    """
    if not BRAVE_API_KEY:
        return "[Brave Search: BRAVE_API_KEY not set]"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                BRAVE_SEARCH_URL,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": BRAVE_API_KEY,
                },
                params={
                    "q": query,
                    "count": count,
                    "search_lang": "en",
                    "country": "AU",
                    "text_decorations": False,
                },
            )
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        return f"[Brave Search error: {e}]"

    results = data.get("web", {}).get("results", [])
    if not results:
        return f"[No results found for: {query}]"

    lines = [f"== Search results: {query} =="]
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("url", "")
        description = r.get("description", "").strip()
        lines.append(f"\n[{i}] {title}")
        lines.append(f"    {url}")
        if description:
            lines.append(f"    {description}")

    return "\n".join(lines)


async def fetch_page(url: str) -> str:
    """
    Fetch a web page and return its main text content.
    Strips navigation, scripts, and boilerplate. Truncates to PAGE_CHAR_LIMIT.
    """
    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return f"[Cannot read page: content type is {content_type}]"
            html = response.text
    except Exception as e:
        return f"[Page fetch error: {e}]"

    try:
        soup = BeautifulSoup(html, "html.parser")

        for tag in STRIP_TAGS:
            for el in soup.find_all(tag):
                el.decompose()

        # Prefer main content areas if present
        main = (
            soup.find("main") or
            soup.find("article") or
            soup.find(id="content") or
            soup.find(class_="content") or
            soup.body
        )

        if not main:
            return "[Could not extract page content]"

        text = main.get_text(separator="\n", strip=True)

        # Collapse excessive blank lines
        lines = [l for l in text.splitlines() if l.strip()]
        text = "\n".join(lines)

        if len(text) > PAGE_CHAR_LIMIT:
            text = text[:PAGE_CHAR_LIMIT] + f"\n\n[... truncated at {PAGE_CHAR_LIMIT} chars]"

        return f"== Page content: {url} ==\n{text}"

    except Exception as e:
        return f"[Page parse error: {e}]"
