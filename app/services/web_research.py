"""Web research for script generation.

Gives *any* LLM access to fresh facts — a local Ollama model with no tool
calling gets the same research as a frontier API model, because the fetching
happens here and the findings are appended to the prompt.

Search backend (``web_search_provider``), both free and keyless:
  - ``duckduckgo`` (default) — HTML endpoint, fine for a few queries per
    script, rate-limited if hammered.
  - ``searxng`` — the meta-search container in docker-compose.yml, which
    aggregates DuckDuckGo/Brave/Wikipedia and is not rate-limited by them.

Pages are read by the Lightpanda headless browser container (CDP over
websocket), so JavaScript-rendered articles produce real text instead of an
empty shell. Plain HTTP is the fallback when the browser is unreachable.

Wikipedia is always queried alongside, since it is keyless, stable and
answers most "what is X" background questions in one request.
"""

import html
import itertools
import json
import os
import re
import time
from urllib.parse import parse_qs, quote_plus, urlsplit

import requests
from loguru import logger

from app.config import config
from app.services import guardrails

DEFAULT_MAX_RESULTS = 6
DEFAULT_FETCH_PAGES = 3
MAX_PAGE_CHARS = 3000
MAX_SNIPPET_CHARS = 400
MAX_WIKIPEDIA_CHARS = 1500
_TIMEOUT = (10, 25)
# Lightpanda renders a page in well under a second; anything slower is a site
# fighting us, and research must not stall a video render.
LIGHTPANDA_TIMEOUT = 20.0
LIGHTPANDA_DEFAULT_URL = "ws://lightpanda:9222"
_LIGHTPANDA_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_MAX_DOWNLOAD_BYTES = 400_000
# A browser that is down must not cost a connect timeout on every page of
# every render, so remember the failure for a short while.
_LIGHTPANDA_RETRY_AFTER = 60.0
_lightpanda_offline_until = 0.0
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)

# Hosts that must never be fetched: a search result (or a redirect) pointing at
# the machine's own network would turn "research the topic" into an SSRF probe.
_PRIVATE_HOST_RE = re.compile(
    r"^(localhost|127\.|0\.|10\.|192\.168\.|169\.254\.|172\.(1[6-9]|2\d|3[01])\.|\[?::1\]?)",
    re.IGNORECASE,
)
_DROP_TAGS_RE = re.compile(
    r"<(script|style|noscript|template|nav|header|footer|aside|form|menu|figure)"
    r"\b.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
# The article body, when the page marks it. Everything outside it is chrome:
# menus, cookie notices, "related stories", newsletter boxes.
_MAIN_CONTENT_RE = re.compile(
    r"<(article|main)\b[^>]*>(?P<body>.*?)</\1>", re.IGNORECASE | re.DOTALL
)
_TAG_RE = re.compile(r"<[^>]+>")
_BLANK_LINES_RE = re.compile(r"\n\s*\n\s*\n+")
# html.duckduckgo.com/html/ markup: title anchor first, snippet anchor after.
_DDG_RESULT_RE = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'(?P<rest>.*?)(?=<a[^>]*class="[^"]*result__a|\Z)',
    re.IGNORECASE | re.DOTALL,
)
_DDG_SNIPPET_RE = re.compile(
    r'class=["\'][^"\']*result__snippet[^"\']*["\'][^>]*>(?P<snippet>.*?)</(?:a|div|span)>',
    re.IGNORECASE | re.DOTALL,
)


def _app(app_config=None):
    return app_config if app_config is not None else config.app


def _tls_verify(app_config=None) -> bool:
    value = _app(app_config).get("tls_verify", True)
    if isinstance(value, str):
        value = value.strip().lower() not in ("0", "false", "no", "off")
    return bool(value)


def is_enabled(app_config=None) -> bool:
    return bool(_app(app_config).get("enable_web_research", False))


def _int_setting(key: str, default: int, app_config=None) -> int:
    try:
        value = int(_app(app_config).get(key, default))
    except (TypeError, ValueError):
        return default
    return max(0, min(value, 10))


def _get(url: str, app_config=None, **kwargs) -> requests.Response:
    headers = {"User-Agent": _USER_AGENT, **kwargs.pop("headers", {})}
    return requests.get(
        url,
        headers=headers,
        proxies=config.proxy,
        verify=_tls_verify(app_config),
        timeout=_TIMEOUT,
        **kwargs,
    )


def _fetchable(url: str) -> bool:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    return bool(
        parsed.scheme in ("http", "https")
        and parsed.hostname
        and parsed.username is None
        and parsed.password is None
        and not _PRIVATE_HOST_RE.match(parsed.hostname)
    )


def _main_content(markup: str) -> str:
    """Return the page's article body when it has one, else the whole markup.

    Site chrome is not research. Handing the model a navigation menu wastes the
    prompt budget and gives it phrases to parrot that have nothing to do with
    the subject.
    """
    candidates = [
        match.group("body") for match in _MAIN_CONTENT_RE.finditer(markup or "")
    ]
    if not candidates:
        return markup or ""
    return max(candidates, key=len)


def _clean(fragment: str, max_chars: int = MAX_SNIPPET_CHARS) -> str:
    text = _DROP_TAGS_RE.sub(" ", _main_content(fragment or ""))
    text = html.unescape(_TAG_RE.sub(" ", text))
    text = _BLANK_LINES_RE.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text[:max_chars].strip()


def _unwrap_ddg_url(href: str) -> str:
    """DuckDuckGo wraps results as //duckduckgo.com/l/?uddg=<encoded target>."""
    if "uddg=" not in href:
        return href if href.startswith("http") else f"https:{href}"
    target = parse_qs(urlsplit(href).query).get("uddg", [""])[0]
    return target or href


def _search_duckduckgo(query: str, max_results: int, app_config=None) -> list[dict]:
    response = _get(
        f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
        app_config=app_config,
    )
    response.raise_for_status()
    results = []
    for match in _DDG_RESULT_RE.finditer(response.text):
        url = _unwrap_ddg_url(html.unescape(match.group("url")))
        if not _fetchable(url):
            continue
        snippet_match = _DDG_SNIPPET_RE.search(match.group("rest"))
        results.append(
            {
                "title": _clean(match.group("title"), 200),
                "url": url,
                "snippet": _clean(snippet_match.group("snippet"))
                if snippet_match
                else "",
            }
        )
        if len(results) >= max_results:
            break
    return results


def _search_searxng(query: str, max_results: int, app_config=None) -> list[dict]:
    base_url = (
        os.environ.get("SEARXNG_URL", "").strip()
        or str(_app(app_config).get("web_search_base_url", "")).strip()
    )
    if not base_url:
        raise ValueError("searxng search needs SEARXNG_URL or app.web_search_base_url")
    response = _get(
        f"{base_url.rstrip('/')}/search",
        app_config=app_config,
        params={"q": query, "format": "json"},
    )
    response.raise_for_status()
    return [
        {
            "title": _clean(item.get("title", ""), 200),
            "url": item.get("url", ""),
            "snippet": _clean(item.get("content", "")),
        }
        for item in response.json().get("results", [])[:max_results]
        if _fetchable(item.get("url", ""))
    ]


def search(
    query: str, max_results: int = DEFAULT_MAX_RESULTS, app_config=None
) -> list[dict]:
    """Return [{title, url, snippet}] for the query, or [] when search fails."""
    provider = (
        os.environ.get("WEB_SEARCH_PROVIDER", "").strip().lower()
        or str(_app(app_config).get("web_search_provider", "")).strip().lower()
        or "duckduckgo"
    )
    backends = {"duckduckgo": _search_duckduckgo, "searxng": _search_searxng}
    backend = backends.get(provider)
    if backend is None:
        logger.warning(f"unknown web_search_provider '{provider}', using duckduckgo")
        provider, backend = "duckduckgo", _search_duckduckgo
    try:
        results = backend(query, max_results, app_config=app_config)
        if results or provider == "duckduckgo":
            return results
        logger.warning(f"{provider} returned no results; retrying on duckduckgo")
    except Exception as e:
        logger.warning(f"web search failed ({provider}): {e}")
        if provider == "duckduckgo":
            return []

    # The self-hosted engine being down or empty must not silently produce a
    # script with no research behind it.
    try:
        return _search_duckduckgo(query, max_results, app_config=app_config)
    except Exception as e:
        logger.warning(f"duckduckgo fallback failed: {e}")
        return []


class LightpandaError(RuntimeError):
    """Raised when the headless browser cannot produce a page."""


def lightpanda_url(app_config=None) -> str:
    """Websocket endpoint of the Lightpanda container.

    The environment variable wins, because docker-compose sets it; then the
    config file, then the compose service name. There is no off switch: an
    unreachable browser already falls back to plain HTTP on its own.
    """
    endpoint = (
        os.environ.get("LIGHTPANDA_URL", "").strip()
        or str(_app(app_config).get("lightpanda_url", "") or "").strip()
        or LIGHTPANDA_DEFAULT_URL
    )
    if endpoint.startswith("http://"):
        endpoint = "ws://" + endpoint[len("http://") :]
    elif endpoint.startswith("https://"):
        endpoint = "wss://" + endpoint[len("https://") :]
    return endpoint.rstrip("/")


def _cdp_page_html(endpoint: str, url: str, timeout: float) -> str:
    """Render one page in Lightpanda over CDP and return its final HTML.

    Speaks the raw protocol rather than pulling in Puppeteer/Playwright: the
    whole exchange is six messages, and a Node runtime or a second browser
    download in the image would cost far more than it saves.
    """
    from websockets.sync.client import connect

    deadline = time.monotonic() + timeout
    counter = itertools.count(1)

    def remaining() -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise LightpandaError(f"timed out after {timeout:.0f}s rendering {url}")
        return left

    with connect(
        endpoint,
        open_timeout=min(timeout, 10),
        close_timeout=5,
        max_size=_LIGHTPANDA_MAX_MESSAGE_BYTES,
    ) as ws:

        def call(method: str, params: dict | None = None, session_id: str = ""):
            message_id = next(counter)
            payload = {"id": message_id, "method": method, "params": params or {}}
            if session_id:
                payload["sessionId"] = session_id
            ws.send(json.dumps(payload))
            while True:
                message = json.loads(ws.recv(timeout=remaining()))
                if message.get("id") != message_id:
                    continue  # CDP event, not our reply
                if "error" in message:
                    raise LightpandaError(f"{method}: {message['error']}")
                return message.get("result") or {}

        def optional(method: str, params: dict | None = None, session_id: str = ""):
            """Call a method the browser may not implement, and shrug it off.

            Lightpanda implements the subset of CDP that Puppeteer's connect
            flow needs, and that subset moves. Only createTarget, attachToTarget,
            navigate and evaluate are actually required to read a page.
            """
            try:
                return call(method, params, session_id)
            except LightpandaError as e:
                logger.debug(f"lightpanda does not support {method}: {e}")
                return {}

        context_id = optional("Target.createBrowserContext").get("browserContextId", "")
        create_params = {"url": "about:blank"}
        if context_id:
            create_params["browserContextId"] = context_id
        target_id = call("Target.createTarget", create_params).get("targetId", "")
        if not target_id:
            raise LightpandaError("browser did not return a target")
        try:
            session_id = call(
                "Target.attachToTarget", {"targetId": target_id, "flatten": True}
            ).get("sessionId", "")
            optional("Page.enable", session_id=session_id)
            call("Page.navigate", {"url": url}, session_id=session_id)

            # Poll readyState instead of waiting on Page.loadEventFired: a page
            # that never fires the event would otherwise hold the connection
            # until the deadline, and this exits as soon as the DOM is done.
            while True:
                state = (
                    optional(
                        "Runtime.evaluate",
                        {"expression": "document.readyState", "returnByValue": True},
                        session_id=session_id,
                    )
                    .get("result", {})
                    .get("value")
                )
                # Only "loading" means the DOM is still being built. Anything
                # else -- complete, interactive, missing, unexpected -- means
                # take the DOM as it is instead of spinning until the deadline.
                if state != "loading":
                    break
                if remaining() < 0.5:
                    break
                time.sleep(0.2)

            evaluated = call(
                "Runtime.evaluate",
                {
                    "expression": "document.documentElement.outerHTML",
                    "returnByValue": True,
                },
                session_id=session_id,
            )
            return str(evaluated.get("result", {}).get("value") or "")
        finally:
            try:
                call("Target.closeTarget", {"targetId": target_id})
                if context_id:
                    call(
                        "Target.disposeBrowserContext",
                        {"browserContextId": context_id},
                    )
            except Exception:
                pass


def _fetch_page_http(url: str, max_chars: int, app_config=None) -> str:
    response = None
    try:
        response = _get(url, app_config=app_config, stream=True)
        response.raise_for_status()
        if "html" not in response.headers.get("Content-Type", "text/html").lower():
            return ""
        # Cap the download instead of the parse: a stray multi-MB page would
        # otherwise be pulled in full only to be truncated afterwards.
        raw = response.raw.read(_MAX_DOWNLOAD_BYTES, decode_content=True) or b""
        return _clean(raw.decode(response.encoding or "utf-8", "replace"), max_chars)
    except Exception as e:
        logger.warning(f"failed to read {url}: {e}")
        return ""
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


def fetch_page(url: str, max_chars: int = MAX_PAGE_CHARS, app_config=None) -> str:
    """Return the readable text of a page, or "" when it cannot be read.

    Lightpanda first, because most news and trend pages render their article
    body in JavaScript and plain HTTP would hand the model an empty shell.
    """
    if not _fetchable(url):
        return ""

    global _lightpanda_offline_until

    endpoint = lightpanda_url(app_config)
    if endpoint and time.monotonic() >= _lightpanda_offline_until:
        try:
            rendered = _cdp_page_html(endpoint, url, LIGHTPANDA_TIMEOUT)
            text = _clean(rendered, max_chars)
            if text:
                return text
            logger.warning(f"lightpanda returned an empty page for {url}")
        except Exception as e:
            _lightpanda_offline_until = time.monotonic() + _LIGHTPANDA_RETRY_AFTER
            logger.warning(
                f"lightpanda ({endpoint}) could not render {url} ({e}); "
                "falling back to plain HTTP"
            )

    return _fetch_page_http(url, max_chars, app_config=app_config)


def wikipedia_summary(query: str, language: str = "", app_config=None) -> str:
    """Search Wikipedia and return the intro of the best matching article."""
    lang = (language or "en").strip().lower().replace("_", "-").split("-")[0] or "en"
    try:
        response = _get(
            f"https://{lang}.wikipedia.org/w/api.php",
            app_config=app_config,
            params={
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "explaintext": "1",
                "exintro": "1",
                "redirects": "1",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": "1",
            },
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = (page.get("extract") or "").strip()
            if extract:
                return extract[:MAX_WIKIPEDIA_CHARS].strip()
    except Exception as e:
        logger.warning(f"wikipedia lookup failed: {e}")
    return ""


def research(
    subject: str,
    language: str = "",
    max_results: int | None = None,
    fetch_pages: int | None = None,
    app_config=None,
) -> str:
    """Research the subject on the web and return a markdown brief ("" if nothing)."""
    subject = (subject or "").strip()
    if not subject:
        return ""

    if max_results is None:
        max_results = _int_setting(
            "web_research_results", DEFAULT_MAX_RESULTS, app_config
        )
    if fetch_pages is None:
        fetch_pages = _int_setting(
            "web_research_pages", DEFAULT_FETCH_PAGES, app_config
        )

    logger.info(f"researching subject on the web: {subject}")
    sections = []

    encyclopedia = wikipedia_summary(subject, language=language, app_config=app_config)
    if encyclopedia:
        sections.append(f"## Background (Wikipedia)\n{encyclopedia}")

    # Ask for more than we keep: the quality filter throws some away, and a
    # brief built from four solid sources beats one built from eight thin ones.
    found = search(subject, max_results=max_results * 2, app_config=app_config)
    results = guardrails.filter_research_results(found)[:max_results]
    if found and not results:
        logger.warning(
            f"every search result was rejected by the quality filter: {subject}"
        )

    pages_read = 0
    for result in results:
        block = [f"### {result['title'] or result['url']}\nSource: {result['url']}"]
        if result["snippet"]:
            block.append(result["snippet"])
        if pages_read < fetch_pages:
            page_text = fetch_page(result["url"], app_config=app_config)
            if page_text and guardrails.is_usable_page(page_text):
                block.append(page_text)
                pages_read += 1
            elif page_text:
                # A paywall stub or a cookie wall is worse than no source at
                # all: it reads like content and says nothing.
                logger.debug(
                    f"research: page has no usable prose, snippet only: {result['url']}"
                )
        sections.append("\n".join(block))

    if not sections:
        logger.warning(f"web research returned nothing for: {subject}")
        return ""

    logger.success(
        f"web research done: wikipedia={bool(encyclopedia)}, "
        f"sources={len(results)}, pages_read={pages_read}"
    )
    return "\n\n".join(sections)
