from typing import Literal
from urllib.parse import urlparse

QualityTier = Literal[1, 2, 3]

# Tier 1: Government, academic, regulatory, and primary research institutions
TIER_1_SUFFIXES = (".gov", ".edu", ".int", ".gov.in", ".nic.in", ".ac.in", ".ac.uk")
TIER_1_DOMAINS = {
    "who.int",
    "worldbank.org",
    "imf.org",
    "iea.org",
    "un.org",
    "oecd.org",
    "bis.org",
    "europa.eu",
    "rbi.org.in",
    "sebi.gov.in",
    "niti.gov.in",
    "sec.gov",
    "nih.gov",
    "cdc.gov",
    "arxiv.org",
    "nature.com",
    "science.org",
    "ieee.org",
}

# Tier 2: Established news agencies, financial press, and industry analysis
TIER_2_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "economist.com",
    "bbc.com",
    "bbc.co.uk",
    "nytimes.com",
    "theguardian.com",
    "cnbc.com",
    "forbes.com",
    "techcrunch.com",
    "wired.com",
    "theverge.com",
    "hbr.org",
    "mckinsey.com",
    "statista.com",
    "wikipedia.org",
    "livemint.com",
    "economictimes.indiatimes.com",
    "thehindu.com",
    "indianexpress.com",
    "business-standard.com",
}


def extract_domain(url: str | None) -> str:
    """Extracts a clean lowercase domain name (without 'www.') from a URL."""
    if not url:
        return ""
    parsed = urlparse(url.strip())
    host = (parsed.netloc or parsed.path.split("/")[0]).lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host.split(":")[0]


def _matches_domain(domain: str, known_domains: set[str]) -> bool:
    """Checks exact domain or subdomain match (e.g., data.worldbank.org -> worldbank.org)."""
    return any(domain == base or domain.endswith(f".{base}") for base in known_domains)


def classify_source_tier(
    url: str | None,
    source_type: Literal["pdf", "web"] = "web",
) -> QualityTier:
    """
    Assigns a reliability tier:
    1 = Official / Academic / Uploaded Primary PDF
    2 = Major News / Financial / Industry Publications
    3 = General Web / Blogs / Forums
    """
    if source_type == "pdf":
        return 1

    domain = extract_domain(url)
    if not domain:
        return 3

    if domain.endswith(TIER_1_SUFFIXES) or _matches_domain(domain, TIER_1_DOMAINS):
        return 1

    if _matches_domain(domain, TIER_2_DOMAINS):
        return 2

    return 3


def get_tier_label(tier: int) -> str:
    labels = {
        1: "Tier 1 · Official / Primary",
        2: "Tier 2 · Major Media / Industry",
        3: "Tier 3 · General Web",
    }
    return labels.get(tier, "Tier 3 · General Web")