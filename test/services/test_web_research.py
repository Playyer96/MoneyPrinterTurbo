import json
import os
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import web_research as wr

DDG_HTML = (
    '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.test%2F1&rut=x">'
    'Title One</a><a class="result__snippet">Snippet &amp; one</a>'
    '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fb.test%2F2">'
    'Title Two</a><a class="result__snippet">Snippet two</a>'
    '<a class="result__a" href="//duckduckgo.com/l/?uddg=http%3A%2F%2F127.0.0.1%2Fadmin">'
    'Local</a><a class="result__snippet">Should be dropped</a>'
)


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


class TestWebResearch(unittest.TestCase):
    def test_rejects_non_public_urls(self):
        self.assertTrue(wr._fetchable("https://example.com/a?b=1"))
        for url in (
            "http://127.0.0.1:8080/admin",
            "http://192.168.1.1/",
            "http://169.254.169.254/latest/meta-data/",
            "file:///etc/passwd",
            "https://user:pw@example.com/",
        ):
            with self.subTest(url=url):
                self.assertFalse(wr._fetchable(url))

    def test_clean_drops_scripts_and_unescapes_entities(self):
        self.assertEqual(
            wr._clean("<script>bad()</script><p>Hello &amp; hi</p>"), "Hello & hi"
        )

    def test_article_body_wins_over_site_chrome(self):
        markup = (
            "<html><body><nav>Home Menu Search Subscribe</nav>"
            "<header>Site name</header>"
            "<article><p>The eruption lasted nine hours.</p></article>"
            "<footer>Cookie notice</footer></body></html>"
        )
        self.assertEqual(wr._clean(markup), "The eruption lasted nine hours.")

    def test_pages_without_an_article_tag_still_produce_text(self):
        markup = "<html><body><div><p>Plain page text.</p></div></body></html>"
        self.assertEqual(wr._clean(markup), "Plain page text.")

    def test_duckduckgo_results_are_unwrapped_and_filtered(self):
        with patch.object(wr, "_get", return_value=_FakeResponse(DDG_HTML)):
            results = wr._search_duckduckgo("anything", max_results=5)

        self.assertEqual(
            [item["url"] for item in results],
            ["https://a.test/1", "https://b.test/2"],
        )
        self.assertEqual(results[0]["title"], "Title One")
        self.assertEqual(results[0]["snippet"], "Snippet & one")

    def test_search_returns_empty_list_when_backend_fails(self):
        with patch.object(wr, "_search_duckduckgo", side_effect=RuntimeError("boom")):
            self.assertEqual(wr.search("anything", app_config={}), [])

    def test_research_combines_wikipedia_and_results(self):
        page = "The eruption began at dawn. " * 12
        with (
            patch.object(wr, "wikipedia_summary", return_value="Background text."),
            patch.object(
                wr,
                "search",
                return_value=[
                    {"title": "T", "url": "https://a.test/1", "snippet": "S"}
                ],
            ),
            patch.object(wr, "fetch_page", return_value=page),
        ):
            brief = wr.research("subject", app_config={})

        self.assertIn("Background text.", brief)
        self.assertIn("https://a.test/1", brief)
        self.assertIn("The eruption began at dawn.", brief)

    def test_research_drops_pages_without_prose(self):
        with (
            patch.object(wr, "wikipedia_summary", return_value=""),
            patch.object(
                wr,
                "search",
                return_value=[
                    {"title": "T", "url": "https://a.test/1", "snippet": "Snippet."}
                ],
            ),
            patch.object(wr, "fetch_page", return_value="We value your privacy"),
        ):
            brief = wr.research("subject", app_config={})

        self.assertIn("Snippet.", brief)
        self.assertNotIn("We value your privacy", brief)

    def test_research_keeps_one_source_per_site(self):
        results = [
            {"title": "A", "url": "https://a.test/1", "snippet": "one"},
            {"title": "B", "url": "https://a.test/2", "snippet": "two"},
            {"title": "C", "url": "https://b.test/1", "snippet": "three"},
        ]
        with (
            patch.object(wr, "wikipedia_summary", return_value=""),
            patch.object(wr, "search", return_value=results),
            patch.object(wr, "fetch_page", return_value=""),
        ):
            brief = wr.research("subject", app_config={})

        self.assertIn("https://a.test/1", brief)
        self.assertNotIn("https://a.test/2", brief)
        self.assertIn("https://b.test/1", brief)

    def test_research_returns_empty_string_when_nothing_found(self):
        with (
            patch.object(wr, "wikipedia_summary", return_value=""),
            patch.object(wr, "search", return_value=[]),
        ):
            self.assertEqual(wr.research("subject", app_config={}), "")


class TestLightpandaClient(unittest.TestCase):
    """Drive the real CDP client against a fake browser speaking the protocol."""

    PAGE_HTML = "<html><body><p>Rendered by the browser.</p></body></html>"

    def _serve(self, handler):
        from websockets.sync.server import serve

        server = serve(handler, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        return f"ws://127.0.0.1:{server.socket.getsockname()[1]}"

    def _cdp_handler(self, connection):
        # An unsolicited event first: the client must skip it and still match
        # its own reply by id.
        connection.send(json.dumps({"method": "Target.targetCreated", "params": {}}))
        for raw in connection:
            request = json.loads(raw)
            method = request["method"]
            if method == "Target.createBrowserContext":
                result = {"browserContextId": "ctx-1"}
            elif method == "Target.createTarget":
                result = {"targetId": "target-1"}
            elif method == "Target.attachToTarget":
                result = {"sessionId": "session-1"}
            elif method == "Runtime.evaluate":
                expression = request["params"]["expression"]
                value = "complete" if "readyState" in expression else self.PAGE_HTML
                result = {"result": {"value": value}}
            else:
                result = {}
            connection.send(json.dumps({"id": request["id"], "result": result}))

    def test_renders_a_page_over_cdp(self):
        endpoint = self._serve(self._cdp_handler)
        html_text = wr._cdp_page_html(endpoint, "https://a.test/article", timeout=10)
        self.assertIn("Rendered by the browser.", html_text)

    def test_unsupported_optional_methods_do_not_break_rendering(self):
        """Lightpanda implements a subset of CDP; only four calls are required."""

        def picky(connection):
            for raw in connection:
                request = json.loads(raw)
                method = request["method"]
                if method in ("Target.createBrowserContext", "Page.enable"):
                    connection.send(
                        json.dumps(
                            {
                                "id": request["id"],
                                "error": {"message": "not implemented"},
                            }
                        )
                    )
                    continue
                if method == "Target.createTarget":
                    self.assertNotIn("browserContextId", request["params"])
                    result = {"targetId": "target-1"}
                elif method == "Target.attachToTarget":
                    result = {"sessionId": "session-1"}
                elif method == "Runtime.evaluate":
                    result = {"result": {"value": self.PAGE_HTML}}
                else:
                    result = {}
                connection.send(json.dumps({"id": request["id"], "result": result}))

        endpoint = self._serve(picky)
        html_text = wr._cdp_page_html(endpoint, "https://a.test/article", timeout=10)
        self.assertIn("Rendered by the browser.", html_text)

    def test_fetch_page_uses_the_browser_when_it_answers(self):
        endpoint = self._serve(self._cdp_handler)
        with patch.object(wr, "lightpanda_url", return_value=endpoint):
            wr._lightpanda_offline_until = 0.0
            text = wr.fetch_page("https://a.test/article")
        self.assertEqual(text, "Rendered by the browser.")

    def test_cdp_errors_fall_back_to_plain_http(self):
        def failing(connection):
            for raw in connection:
                request = json.loads(raw)
                connection.send(
                    json.dumps(
                        {"id": request["id"], "error": {"message": "not supported"}}
                    )
                )

        endpoint = self._serve(failing)
        with (
            patch.object(wr, "lightpanda_url", return_value=endpoint),
            patch.object(wr, "_fetch_page_http", return_value="plain http text"),
        ):
            wr._lightpanda_offline_until = 0.0
            text = wr.fetch_page("https://a.test/article")
        self.assertEqual(text, "plain http text")

    def test_a_dead_browser_is_skipped_until_the_retry_window_passes(self):
        with (
            patch.object(wr, "lightpanda_url", return_value="ws://127.0.0.1:1"),
            patch.object(wr, "_fetch_page_http", return_value="plain http text"),
            patch.object(wr, "_cdp_page_html") as fake_cdp,
        ):
            fake_cdp.side_effect = OSError("connection refused")
            wr._lightpanda_offline_until = 0.0
            wr.fetch_page("https://a.test/1")
            wr.fetch_page("https://a.test/2")

        self.assertEqual(fake_cdp.call_count, 1)
        wr._lightpanda_offline_until = 0.0

    def test_endpoint_prefers_the_environment_then_config(self):
        with patch.dict(os.environ, {"LIGHTPANDA_URL": "http://browser:9222"}):
            self.assertEqual(wr.lightpanda_url({}), "ws://browser:9222")
        with patch.dict(os.environ, {"LIGHTPANDA_URL": ""}):
            self.assertEqual(
                wr.lightpanda_url({"lightpanda_url": "ws://elsewhere:9222"}),
                "ws://elsewhere:9222",
            )
            self.assertEqual(wr.lightpanda_url({}), wr.LIGHTPANDA_DEFAULT_URL)


class TestSearchFallback(unittest.TestCase):
    def test_searxng_failure_falls_back_to_duckduckgo(self):
        ddg_results = [{"title": "T", "url": "https://a.test/1", "snippet": "S"}]
        with (
            patch.dict(os.environ, {"WEB_SEARCH_PROVIDER": "searxng"}),
            patch.object(wr, "_search_searxng", side_effect=RuntimeError("down")),
            patch.object(wr, "_search_duckduckgo", return_value=ddg_results),
        ):
            self.assertEqual(wr.search("anything", app_config={}), ddg_results)

    def test_empty_searxng_result_falls_back_to_duckduckgo(self):
        ddg_results = [{"title": "T", "url": "https://a.test/1", "snippet": "S"}]
        with (
            patch.dict(os.environ, {"WEB_SEARCH_PROVIDER": "searxng"}),
            patch.object(wr, "_search_searxng", return_value=[]),
            patch.object(wr, "_search_duckduckgo", return_value=ddg_results),
        ):
            self.assertEqual(wr.search("anything", app_config={}), ddg_results)


if __name__ == "__main__":
    unittest.main()
