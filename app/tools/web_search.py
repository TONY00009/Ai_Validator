import logging
from typing import Literal

from bs4 import BeautifulSoup
from ddgs import DDGS
import httpx
from pydantic import BaseModel
from tavily import TavilyClient

from app.core.config import get_settings
from app.tools.source_tiers import classify_source_tier, extract_domain

logger = logging.getLogger(__name__)


class WebSearchResult(BaseModel):
    title: str
    url: str
    domain: str
    quality_tier: int
    content: str
    provider: Literal["tavily", "ddgs"]


def fetch_page_text(url: str, timeout: float = 8.0, max_chars: int = 8000) -> str:
    """
    Lightweight fallback HTML reader: downloads a webpage and strips scripts,
    menus, and ads so only readable article text remains.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "").lower()
        if "text/html" not in content_type and "text/plain" not in content_type:
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        return text[:max_chars]
    except Exception as exc:
        logger.debug("Page fetch skipped for %s: %s", url, exc)
        return ""


def _search_tavily(query: str, max_results: int) -> list[WebSearchResult]:
    settings = get_settings()
    if not settings.tavily_api_key:
        raise ValueError("TAVILY_API_KEY is missing in .env")

    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",
        include_raw_content=False,
    )

    results: list[WebSearchResult] = []
    for item in response.get("results", []):
        url = (item.get("url") or "").strip()
        content = (item.get("content") or "").strip()
        if not url or not content:
            continue
        results.append(
            WebSearchResult(
                title=(item.get("title") or url).strip(),
                url=url,
                domain=extract_domain(url),
                quality_tier=classify_source_tier(url, source_type="web"),
                content=content,
                provider="tavily",
            )
        )
    return results


def _search_ddgs(query: str, max_results: int, enrich_short: bool = True) -> list[WebSearchResult]:
    raw_items = DDGS().text(query, max_results=max_results) or []
    results: list[WebSearchResult] = []

    for item in raw_items:
        url = (item.get("href") or "").strip()
        snippet = (item.get("body") or "").strip()
        if not url:
            continue

        content = snippet
        # DuckDuckGo snippets are often 1-2 lines; try fetching fuller article text
        if enrich_short and len(content) < 350:
            fetched = fetch_page_text(url)
            if len(fetched) > len(content):
                content = fetched

        if not content:
            continue

        results.append(
            WebSearchResult(
                title=(item.get("title") or url).strip(),
                url=url,
                domain=extract_domain(url),
                quality_tier=classify_source_tier(url, source_type="web"),
                content=content,
                provider="ddgs",
            )
        )
    return results


def search_web(query: str, max_results: int = 5) -> list[WebSearchResult]:
    """
    Primary: Tavily Search (AI-ready clean content).
    Fallback: DuckDuckGo Search + HTML page enrichment.
    Results are deduplicated by URL and sorted so Tier 1 sources appear first.
    """
    results: list[WebSearchResult] = []

    try:
        results = _search_tavily(query, max_results=max_results)
    except Exception as exc:
        logger.warning("⚠️  Tavily search failed (%s). Falling back to DuckDuckGo...", exc)

    if not results:
        try:
            results = _search_ddgs(query, max_results=max_results)
        except Exception as exc:
            logger.error("❌ DuckDuckGo fallback also failed: %s", exc)
            return []

    # Deduplicate by URL and sort by quality_tier (Tier 1 first)
    seen_urls: set[str] = set()
    unique: list[WebSearchResult] = []
    for item in results:
        if item.url not in seen_urls:
            seen_urls.add(item.url)
            unique.append(item)

    unique.sort(key=lambda r: r.quality_tier)
    return unique