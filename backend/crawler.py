"""
Live crawler for Siddhanta Knowledge websites.

This module is intentionally dependency-light (requests + BeautifulSoup, no AWS,
no Chroma) so it can be run and tested on its own:

    python -m crawler            # crawl live sites and rewrite data/*.json

It regenerates two files in the data directory:

  * website.json          - one entry per important page + one per live course.
                            Canonical ids used by main.py hard-coded handlers
                            (pricing-information, refund-policy, aajivan-overview,
                            course-*, ...) are preserved so those answers keep
                            working; new pages (blogs, events, prakashan,
                            sandhaan/shodha sub-pages, siddhantavijnan) are added.
  * courses_catalog.json  - the authoritative live list of enrollable courses on
                            Siksha, used for an accurate, dynamic course count.

Nothing here writes to Chroma. The refresh orchestration (crawl + re-embed) lives
in refresh.py so this file stays safe to import and run anywhere.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import time
from datetime import date
from typing import Iterable, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("siddh_guide.crawler")

MAIN_SITE = "https://siddhantaknowledge.org"
SIKSHA_SITE = "https://siksha.siddhantaknowledge.org"
VIJNAN_SITE = "https://siddhantavijnan.org"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SiddhGuideBot/1.0; +https://siddhantaknowledge.org)"
}
REQUEST_TIMEOUT = int(os.getenv("CRAWL_TIMEOUT", "30"))
REQUEST_PAUSE = float(os.getenv("CRAWL_PAUSE_SECONDS", "0.4"))
MAX_CONTENT_CHARS = int(os.getenv("CRAWL_MAX_CONTENT_CHARS", "1800"))
MAX_BLOG_POSTS = int(os.getenv("CRAWL_MAX_BLOG_POSTS", "15"))

# Synthetic entries that main.py answers with hard-coded prose keyed by id, but
# which do not map to a single crawlable URL. They are carried over verbatim from
# the previous website.json so those answers (and their tests) keep resolving.
PRESERVE_IDS = {"pricing-information", "enrollment-overview", "about-what-siddhanta-does"}

# Slugs that live under the WooCommerce "product" post type but are not courses.
NON_COURSE_PRODUCT_SLUGS = {"eie-quiz", "gift-coupon"}
NON_COURSE_SLUG_MARKERS = ("quiz", "coupon", "gift", "-demo", "demo-", "survey")

# Shared search keywords so "who are the research team members / researchers"
# retrieves the Shodha research pages (whose team names are pulled live by the
# crawler). Hints only — the actual member names come from the live page.
_RESEARCH_TEAM_KEYWORDS = [
    "research team", "research team members", "team members", "researchers",
    "who are the researchers", "who is on the research team", "research members",
]

# Curated pages to crawl. The id is stable; where main.py relies on a specific id
# for a hard-coded answer, that id is reused so the answer keeps resolving.
CURATED_PAGES: list[dict] = [
    {"id": "home-siddhanta-overview", "url": f"{MAIN_SITE}/", "category": "about", "title": "Siddhanta Knowledge Foundation"},
    {"id": "about-siddhanta", "url": f"{MAIN_SITE}/about-us/", "category": "about", "title": "About Siddhanta"},
    {"id": "siksha-platform-overview", "url": f"{SIKSHA_SITE}/", "category": "platform", "title": "Siksha"},
    {"id": "aajivan-overview", "url": f"{MAIN_SITE}/aajivan/", "category": "course", "title": "Aajivan"},
    {"id": "sandhaan-technology-overview", "url": f"{MAIN_SITE}/sandhaan/", "category": "platform", "title": "Sandhaan"},
    {"id": "siddhanta-kosha-overview", "url": f"{MAIN_SITE}/sandhaan/siddhanta-kosha/", "category": "platform", "title": "Siddhanta Kosha"},
    {"id": "shastra-maps-overview", "url": f"{MAIN_SITE}/sandhaan/shastra-maps/", "category": "platform", "title": "Shastra Maps"},
    {"id": "sandhaan-linguistics", "url": f"{MAIN_SITE}/sandhaan/linguistics/", "category": "research", "title": "Sandhaan - Linguistics"},
    {"id": "sandhaan-jyotisha", "url": f"{MAIN_SITE}/sandhaan/jyotisha/", "category": "research", "title": "Sandhaan - Jyotisha"},
    {"id": "sandhaan-yoga", "url": f"{MAIN_SITE}/sandhaan/yoga/", "category": "research", "title": "Sandhaan - Yoga"},
    {"id": "shodha-research-overview", "url": f"{MAIN_SITE}/shodha/", "category": "about", "title": "Shodha", "keywords": _RESEARCH_TEAM_KEYWORDS},
    {"id": "shodha-siddhanta-prastuti", "url": f"{MAIN_SITE}/shodha/siddhanta-prastuti/", "category": "research", "title": "Shodha - Siddhanta Prastuti", "keywords": _RESEARCH_TEAM_KEYWORDS},
    {"id": "shodha-indic-thought-models", "url": f"{MAIN_SITE}/shodha/indic-thought-models/", "category": "research", "title": "Shodha - Indic Thought Models", "keywords": _RESEARCH_TEAM_KEYWORDS},
    {"id": "shodha-conscious-enterprise-management", "url": f"{MAIN_SITE}/shodha/conscious-enterprise-management/", "category": "research", "title": "Shodha - Conscious Enterprise Management", "keywords": _RESEARCH_TEAM_KEYWORDS},
    {"id": "shodha-transformational-bharatiya-pedagogy", "url": f"{MAIN_SITE}/shodha/transformational-bharatiya-pedagogy/", "category": "research", "title": "Shodha - Transformational Bharatiya Pedagogy", "keywords": _RESEARCH_TEAM_KEYWORDS},
    {"id": "prakashan-overview", "url": f"{MAIN_SITE}/prakashan/", "category": "publication", "title": "Prakashan (Publications)",
     "keywords": ["publications", "publication", "books", "book", "prakashan",
                  "buy books", "purchase books", "purchase publications",
                  "where to buy", "list of books", "books published", "our books"]},
    {"id": "events-overview", "url": f"{MAIN_SITE}/events/", "category": "events", "title": "Events"},
    {"id": "siddhantavijnan-overview", "url": f"{VIJNAN_SITE}/", "category": "platform", "title": "Siddhanta Vijnan"},
    {"id": "contact-support", "url": f"{MAIN_SITE}/contact/", "category": "support", "title": "Contact Siddhanta"},
    {"id": "refund-policy", "url": f"{MAIN_SITE}/refund-policy/", "category": "policy", "title": "Refund Policy"},
    {"id": "privacy-policy", "url": f"{MAIN_SITE}/privacy-policy/", "category": "policy", "title": "Privacy Policy"},
    {"id": "terms-of-use-overview", "url": f"{MAIN_SITE}/terms-of-use/", "category": "policy", "title": "Terms of Use"},
    {"id": "copyright-policy", "url": f"{MAIN_SITE}/copyright-policy/", "category": "policy", "title": "Copyright Policy"},
]

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(HEADERS)
    return _session


def _clean_text(raw: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(raw or "")).strip()


def fetch_html(url: str) -> Optional[str]:
    try:
        resp = _get_session().get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            logger.warning("crawl: %s returned %s", url, resp.status_code)
            return None
        return resp.text
    except Exception:
        logger.warning("crawl: failed to fetch %s", url, exc_info=True)
        return None
    finally:
        time.sleep(REQUEST_PAUSE)


def fetch_json(url: str, params: Optional[dict] = None) -> Optional[object]:
    try:
        resp = _get_session().get(url, params=params or {}, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        logger.warning("crawl: failed to fetch json %s", url, exc_info=True)
        return None
    finally:
        time.sleep(REQUEST_PAUSE)


def extract_page_text(page_html: str, max_chars: int = MAX_CONTENT_CHARS) -> str:
    """Strip chrome and return readable body text from an arbitrary WP page."""
    soup = BeautifulSoup(page_html, "html.parser")

    for tag in soup(
        ["script", "style", "noscript", "nav", "header", "footer", "form", "svg", "iframe"]
    ):
        tag.decompose()

    # Elementor / WP themes: prefer the main article/content region if present.
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"class": re.compile(r"(entry-content|site-content|elementor)")})
        or soup.body
        or soup
    )

    # Collect section / item headings first (e.g. the "Our Books" titles on the
    # Publications page, event names, etc.). They are pulled live from the page —
    # nothing hardcoded — and placed at the front so they survive the length cap
    # and are embedded prominently. This is what lets the bot list the actual
    # publications/books dynamically instead of only the page's intro text.
    headings: list[str] = []
    for h in main.find_all(["h1", "h2", "h3", "h4"]):
        ht = _clean_text(h.get_text(" ", strip=True))
        if 3 <= len(ht) <= 140 and ht not in headings:
            headings.append(ht)

    body = _clean_text(main.get_text(separator=" ", strip=True))
    text = (" | ".join(headings) + ". " + body).strip() if headings else body

    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


def _keywords_from(title: str, extra: Iterable[str] = ()) -> list[str]:
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", title)]
    seen: list[str] = []
    for token in [title, *words, *extra]:
        token = token.strip()
        if token and token not in seen:
            seen.append(token)
    return seen[:12]


# --------------------------------------------------------------------------- #
# Courses (WooCommerce products on Siksha)
# --------------------------------------------------------------------------- #

def _is_real_course(slug: str) -> bool:
    slug = (slug or "").lower()
    if slug in NON_COURSE_PRODUCT_SLUGS:
        return False
    return not any(marker in slug for marker in NON_COURSE_SLUG_MARKERS)


def _product_categories(product: dict) -> list[str]:
    names: list[str] = []
    for group in (product.get("_embedded", {}) or {}).get("wp:term", []) or []:
        for term in group or []:
            name = (term.get("name") or "").strip()
            taxonomy = term.get("taxonomy", "")
            if name and taxonomy == "product_cat":
                names.append(name)
    return names


def _extract_price(page_html: str) -> Optional[str]:
    soup = BeautifulSoup(page_html, "html.parser")
    amounts: list[str] = []
    for span in soup.select(".woocommerce-Price-amount"):
        val = _clean_text(span.get_text(" ", strip=True))
        if val:
            amounts.append(val)

    def numeric(v: str) -> float:
        digits = re.sub(r"[^\d.]", "", v.replace(",", ""))
        try:
            return float(digits) if digits else 0.0
        except ValueError:
            return 0.0

    non_zero = [a for a in amounts if numeric(a) > 0]
    if non_zero:
        # last non-zero price is the current/sale price in WooCommerce markup
        return non_zero[-1]
    return amounts[0] if amounts else None


def _extract_course_meta(page_text: str) -> dict:
    meta: dict = {}
    dur = re.search(r"(\d+)\s*Hours?\b", page_text, re.IGNORECASE)
    if dur:
        meta["duration"] = f"{dur.group(1)} Hours"
    aud = re.search(r"\b(UG\s*/\s*PG|UG\s*&\s*PG|UG|PG|School|Professionals?)\b", page_text)
    if aud:
        meta["audience"] = _clean_text(aud.group(1)).replace(" ", "")
    return meta


def _visible_course_slugs() -> set[str]:
    """Slugs of products actually shown on the public Siksha shop.

    The WooCommerce shop reflects catalog visibility (it hides upcoming and
    catalog-excluded products), so it is the source of truth for the count a
    visitor sees — the REST product list includes hidden ones and over-counts.
    """
    slugs: set[str] = set()
    url = f"{SIKSHA_SITE}/shop/"
    pages = 0
    link_re = re.compile(
        r'href="(https://siksha\.siddhantaknowledge\.org/[a-z0-9\-]+/)"'
        r'[^>]*class="[^"]*woocommerce-LoopProduct-link'
    )
    next_re = re.compile(r'<a[^>]*class="[^"]*next[^"]*"[^>]*href="([^"]+)"')

    while url and pages < 15:
        page_html = fetch_html(url)
        pages += 1
        if not page_html:
            break
        for link in link_re.findall(page_html):
            slugs.add(link.rstrip("/").rsplit("/", 1)[-1])
        m = next_re.search(page_html)
        url = html.unescape(m.group(1)) if m else None

    logger.info("crawl: %s catalog-visible course slugs found on shop", len(slugs))
    return slugs


def crawl_courses() -> tuple[list[dict], list[dict]]:
    """Return (website_course_entries, catalog_entries) for live Siksha courses."""
    products: list[dict] = []
    page = 1
    while True:
        batch = fetch_json(
            f"{SIKSHA_SITE}/wp-json/wp/v2/product",
            params={"per_page": 100, "page": page, "_embed": 1},
        )
        if not batch:
            break
        products.extend(batch)
        if len(batch) < 100:
            break
        page += 1

    # Restrict to courses a visitor can actually see in the shop. If the shop
    # scrape fails we keep all real courses rather than returning nothing.
    visible_slugs = _visible_course_slugs()

    website_entries: list[dict] = []
    catalog: list[dict] = []
    today = date.today().isoformat()

    for product in products:
        slug = (product.get("slug") or "").strip()
        title = _clean_text((product.get("title") or {}).get("rendered", ""))
        link = (product.get("link") or "").strip()
        if not slug or not title or not _is_real_course(slug):
            continue
        # Only keep courses visible in the public shop (when we have that list).
        if visible_slugs and slug not in visible_slugs:
            continue

        categories = _product_categories(product)
        description = extract_page_text(
            (product.get("content") or {}).get("rendered", "")
            or (product.get("excerpt") or {}).get("rendered", ""),
            max_chars=900,
        )

        price = None
        page_html = fetch_html(link)
        page_text = extract_page_text(page_html, max_chars=4000) if page_html else ""
        if page_html:
            price = _extract_price(page_html)
        course_meta = _extract_course_meta(page_text)

        catalog.append(
            {
                "title": title,
                "slug": slug,
                "url": link,
                "categories": categories,
                "price": price,
                "duration": course_meta.get("duration"),
                "audience": course_meta.get("audience"),
                # publish/modified dates drive "latest course launched" answers
                "published": (product.get("date") or "")[:10],
                "modified": (product.get("modified") or "")[:10],
            }
        )

        parts = [f"Course title: {title}."]
        if course_meta.get("duration"):
            parts.append(f"Duration: {course_meta['duration']}.")
        if course_meta.get("audience"):
            parts.append(f"Applicable audience: {course_meta['audience']}.")
        if categories:
            parts.append(f"Categories shown: {', '.join(categories)}.")
        if price:
            parts.append(f"Price shown: {price}.")
        if description:
            parts.append(description)
        content = " ".join(parts)
        if len(content) > MAX_CONTENT_CHARS:
            content = content[:MAX_CONTENT_CHARS].rsplit(" ", 1)[0] + "..."

        website_entries.append(
            {
                "id": f"course-{slug}",
                "title": title,
                "source_url": link,
                "category": "course",
                "keywords": _keywords_from(title, categories),
                "content": content,
                "last_verified": today,
            }
        )

    logger.info("crawl: %s live courses collected", len(catalog))
    return website_entries, catalog


# --------------------------------------------------------------------------- #
# Blogs
# --------------------------------------------------------------------------- #

def crawl_blogs() -> list[dict]:
    posts = fetch_json(
        f"{MAIN_SITE}/wp-json/wp/v2/posts",
        params={"per_page": MAX_BLOG_POSTS, "_fields": "title,excerpt,link,date"},
    )
    if not posts:
        return []

    today = date.today().isoformat()
    entries: list[dict] = []
    titles: list[str] = []

    for post in posts:
        title = _clean_text((post.get("title") or {}).get("rendered", ""))
        excerpt = extract_page_text((post.get("excerpt") or {}).get("rendered", ""), max_chars=600)
        link = (post.get("link") or "").strip()
        posted = (post.get("date") or "")[:10]
        if not title:
            continue
        titles.append(f"{title} ({posted})" if posted else title)
        entries.append(
            {
                "id": f"blog-{re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')[:60]}",
                "title": title,
                "source_url": link,
                "category": "blog",
                "keywords": _keywords_from(title, ["blog", "article"]),
                "content": f"Blog post: {title}. Published {posted}. {excerpt}".strip(),
                "last_verified": today,
            }
        )

    if titles:
        entries.insert(
            0,
            {
                "id": "blogs-overview",
                "title": "Siddhanta Blogs",
                "source_url": f"{MAIN_SITE}/blogs/",
                "category": "blog",
                "keywords": ["blog", "blogs", "articles", "posts"],
                "content": (
                    f"The Siddhanta Knowledge Foundation blog currently lists "
                    f"{len(titles)} recent posts: " + "; ".join(titles) + "."
                ),
                "last_verified": today,
            },
        )
    logger.info("crawl: %s blog entries collected", len(entries))
    return entries


# --------------------------------------------------------------------------- #
# Curated pages
# --------------------------------------------------------------------------- #

def crawl_curated_pages() -> list[dict]:
    today = date.today().isoformat()
    entries: list[dict] = []
    for page in CURATED_PAGES:
        page_html = fetch_html(page["url"])
        if not page_html:
            continue
        content = extract_page_text(page_html)
        if not content or len(content) < 40:
            continue
        entries.append(
            {
                "id": page["id"],
                "title": page["title"],
                "source_url": page["url"],
                "category": page["category"],
                "keywords": _keywords_from(page["title"], page.get("keywords", [])),
                "content": content,
                "last_verified": today,
            }
        )
    logger.info("crawl: %s curated page entries collected", len(entries))
    return entries


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def _load_existing_entries(data_dir: Optional[str]) -> list[dict]:
    if not data_dir:
        return []
    path = os.path.join(data_dir, "website.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        logger.warning("crawl: could not read existing website.json for preserve step", exc_info=True)
        return []


def build_dataset(preserve_dir: Optional[str] = None) -> tuple[list[dict], list[dict]]:
    """Crawl everything and return (website_entries, course_catalog).

    Synthetic PRESERVE_IDS entries (hard-coded answers in main.py) are carried
    over from the existing website.json in ``preserve_dir`` so they never vanish.
    """
    website_entries: list[dict] = []
    website_entries.extend(crawl_curated_pages())

    course_entries, catalog = crawl_courses()
    website_entries.extend(course_entries)
    website_entries.extend(crawl_blogs())

    # de-duplicate by id, last one wins
    by_id: dict[str, dict] = {}
    for entry in website_entries:
        by_id[entry["id"]] = entry

    for entry in _load_existing_entries(preserve_dir):
        eid = (entry.get("id") or "").strip()
        if eid in PRESERVE_IDS and eid not in by_id:
            by_id[eid] = entry

    return list(by_id.values()), catalog


def _atomic_write(path: str, payload: object) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def crawl_to_files(data_dir: str) -> dict:
    """Crawl and rewrite website.json + courses_catalog.json in data_dir.

    Returns a summary dict. On a failed/empty crawl the existing files are left
    untouched so the service always keeps working from the last-good snapshot.
    """
    website_entries, catalog = build_dataset(preserve_dir=data_dir)

    if len(website_entries) < 5:
        logger.error("crawl produced too few entries (%s); keeping existing files", len(website_entries))
        return {"ok": False, "website_entries": len(website_entries), "courses": len(catalog)}

    website_path = os.path.join(data_dir, "website.json")
    catalog_path = os.path.join(data_dir, "courses_catalog.json")

    _atomic_write(website_path, website_entries)
    if catalog:
        _atomic_write(catalog_path, catalog)

    summary = {
        "ok": True,
        "website_entries": len(website_entries),
        "courses": len(catalog),
        "generated": date.today().isoformat(),
    }
    logger.info("crawl_to_files: wrote %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    here = os.path.dirname(os.path.abspath(__file__))
    print(json.dumps(crawl_to_files(os.path.join(here, "data")), indent=2))
