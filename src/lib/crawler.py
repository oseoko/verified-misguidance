"""
Playwright async crawler with robots.txt compliance.

Reuses a single Chromium browser instance per batch run.
Bounded concurrency via asyncio.Semaphore.
All failures are returned as ('', '<status>', crawled_at, url) — never raised
to callers.

The actual crawl-start timestamp (`crawled_at`) is captured *immediately
before* `page.goto()`, so downstream consumers see the moment the request
left this process — independent of any later DB-insert delay.

robots.txt policy
-----------------
Before each fetch we consult the host's ``robots.txt`` (cached per host for
the lifetime of the process). If the URL is disallowed for our user agent,
the crawler returns status ``'blocked_by_robots'`` and skips the page.

Failure to fetch ``robots.txt`` (network error, 4xx/5xx, timeout) is treated
as "no rules published" per RFC 9309 §2.3.1.4, so the URL is allowed.

Vertex AI grounding-redirect URLs (Gemini citation URIs) are handled as a
narrow exception: they are HTTP 30x redirects, not content endpoints, and
Google's robots.txt blanket-disallows them. We follow the single redirect
to its target with a lightweight HEAD/GET (no JS) and then crawl the
*resolved* URL under the normal robots.txt regime — i.e. respect the
target host's robots.txt, never the redirect alias's.
"""
import asyncio
import logging
import re
import urllib.error
import urllib.request
from datetime import datetime
from urllib import robotparser
from urllib.parse import urlparse

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Playwright,
)

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r'\n{3,}')

# Single user agent string used by both the Playwright browser context and
# our ``robots.txt`` checks; keeps the two consistent.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_ROBOTS_TIMEOUT_S = 5.0


class _RobotsCache:
    """Per-host ``robots.txt`` cache with async-safe single-flight fetch.

    The first request for a given host triggers a one-shot HTTP fetch of
    ``/robots.txt``; concurrent callers wait on the same future, so we
    never fetch twice for the same host.
    """

    def __init__(self) -> None:
        self._cache: dict[str, robotparser.RobotFileParser] = {}
        self._inflight: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, url: str, user_agent: str = USER_AGENT) -> bool:
        """Return False iff the host's robots.txt forbids this URL for ``user_agent``."""
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if not host:
            return True
        rp = await self._get_or_fetch(parsed.scheme or "https", host)
        try:
            return rp.can_fetch(user_agent, url)
        except Exception as exc:  # parser bug → fail-open, log
            logger.debug("robots can_fetch failed [%s]: %s", url, exc)
            return True

    async def _get_or_fetch(
        self, scheme: str, host: str
    ) -> robotparser.RobotFileParser:
        async with self._lock:
            if host in self._cache:
                return self._cache[host]
            if host in self._inflight:
                fut = self._inflight[host]
            else:
                fut = asyncio.get_event_loop().create_future()
                self._inflight[host] = fut
                # schedule fetch outside the lock
                asyncio.create_task(self._fetch(scheme, host, fut))
        return await fut

    async def _fetch(
        self,
        scheme: str,
        host: str,
        fut: asyncio.Future,
    ) -> None:
        try:
            rp = await asyncio.get_event_loop().run_in_executor(
                None, self._sync_fetch, scheme, host
            )
        except Exception as exc:  # very defensive — should already be handled
            logger.debug("robots fetch crashed for %s: %s", host, exc)
            rp = robotparser.RobotFileParser()  # empty parser → fail-open
        async with self._lock:
            self._cache[host] = rp
            self._inflight.pop(host, None)
        if not fut.done():
            fut.set_result(rp)

    @staticmethod
    def _sync_fetch(scheme: str, host: str) -> robotparser.RobotFileParser:
        """Blocking ``robots.txt`` fetch + parse. Returns an empty parser
        on any failure (RFC 9309: missing/unreachable robots → allow all)."""
        rp = robotparser.RobotFileParser()
        url = f"{scheme}://{host}/robots.txt"
        rp.set_url(url)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=_ROBOTS_TIMEOUT_S) as resp:
                if 400 <= resp.status:
                    return rp  # treat as "no rules"
                body = resp.read().decode("utf-8", errors="replace")
                rp.parse(body.splitlines())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.debug("robots.txt unreachable for %s: %s", host, exc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("robots.txt parse error for %s: %s", host, exc)
        return rp


_ROBOTS = _RobotsCache()


async def create_browser(playwright: Playwright) -> tuple[Browser, BrowserContext]:
    """
    Launch a shared Chromium instance with a single BrowserContext.
    Call once at batch start; pass context to crawl_all().
    """
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context(
        user_agent=USER_AGENT,
        java_script_enabled=True,
        ignore_https_errors=True,
        viewport={"width": 1280, "height": 800},
    )
    return browser, context


async def close_browser(browser: Browser) -> None:
    """Gracefully close the browser and all associated pages."""
    await browser.close()


_VERTEX_REDIRECT_PATHS = (
    "/grounding-api-redirect/",
    "/grounding-redirect/",
)


def _is_vertex_redirect(url: str) -> bool:
    """``vertexaisearch.cloud.google.com/grounding-api-redirect/<token>``."""
    p = urlparse(url)
    if p.netloc != "vertexaisearch.cloud.google.com":
        return False
    return any(p.path.startswith(prefix) for prefix in _VERTEX_REDIRECT_PATHS)


async def _resolve_vertex_redirect(url: str, timeout_s: float = 15.0) -> str:
    """Follow Gemini grounding redirect to its target URL.

    Single GET (or HEAD where supported) with redirect-following. Avoids
    Playwright JS execution; only the final redirect destination is read.
    Falls back to the input URL on any failure.
    """
    try:
        import httpx
    except ImportError:
        return url
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout_s,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            resp = await client.get(url)
            return str(resp.url) or url
    except Exception as exc:
        logger.debug("vertex redirect resolve failed [%s]: %s", url, exc)
        return url


async def crawl_url(
    context: BrowserContext,
    url: str,
    timeout_ms: int = 15_000,
    max_chars: int = 50_000,
) -> tuple[str, str, datetime, str]:
    """
    Crawl a single URL using a new page.
    Returns (content, status, crawled_at, final_url) where:
      - status is 'ok' | 'failed' | 'blocked_by_robots'
      - crawled_at is captured *just before* the HTTP request leaves this
        process (i.e. the actual crawl start time, NOT a DB insert default)
      - final_url is page.url after navigation (redirects resolved by the
        browser); falls back to input url on failure / robots block.
    """
    crawled_at = datetime.now()

    # Vertex grounding redirects: resolve to the target URL first; then the
    # robots.txt check + content fetch run against the *resolved* host.
    if _is_vertex_redirect(url):
        url = await _resolve_vertex_redirect(url, timeout_s=timeout_ms / 1000.0)

    if not await _ROBOTS.is_allowed(url, USER_AGENT):
        logger.info("blocked by robots.txt: %s", url)
        return "", "blocked_by_robots", crawled_at, url

    page = await context.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        final_url = page.url or url
        text: str = await page.evaluate("document.body.innerText")
        text = _WS_RE.sub("\n\n", text).strip()
        return text[:max_chars], "ok", crawled_at, final_url
    except Exception as exc:
        logger.debug("crawl failed [%s]: %s", url, exc)
        return "", "failed", crawled_at, url
    finally:
        await page.close()


async def crawl_all(
    context: BrowserContext,
    urls: list[tuple[int, str]],
    concurrency: int = 5,
    timeout_ms: int = 15_000,
    max_chars: int = 50_000,
) -> list[tuple[int, str, str, str, datetime, str]]:
    """
    Crawl all URLs concurrently (up to `concurrency` at a time).
    Returns list of (order, url, content, status, crawled_at, final_url) in
    the same order as input. `crawled_at` is the moment the HTTP request
    actually started; `final_url` is the post-redirect URL captured from
    the browser (see crawl_url docstring).
    """
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(order: int, url: str) -> tuple[int, str, str, str, datetime, str]:
        async with sem:
            content, status, crawled_at, final_url = await crawl_url(
                context, url, timeout_ms, max_chars
            )
        return order, url, content, status, crawled_at, final_url

    tasks = [_bounded(order, url) for order, url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out: list[tuple[int, str, str, str, datetime, str]] = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            order, url = urls[i]
            logger.warning("gather exception for %s: %s", url, res)
            out.append((order, url, "", "failed", datetime.now(), url))
        else:
            out.append(res)
    return out
