import ast
import json
import re
import unittest
from pathlib import Path

from app.models.llm_provider import get_llm_provider
from app.utils import utils


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"
I18N_DIR = ROOT_DIR / "webui" / "i18n"
LLM_PROVIDER_TIPS_PREFIX = "llm_provider_tips."
TTS_PROVIDER_TIPS_PREFIX = "tts_provider_tips."
SECONDARY_LOCALES = ("de", "es", "fr", "id", "it", "ko", "pt", "ru", "tr", "vi")
PROVIDER_TIPS_PREFIXES = (
    LLM_PROVIDER_TIPS_PREFIX,
    TTS_PROVIDER_TIPS_PREFIX,
)
# Partner provider brand names and long descriptions are only maintained in
# Chinese and English. Secondary locales fall back to English uniformly,
# avoiding copying identical brand names ten times and preventing long
# descriptions from only being updated in some languages.
ENGLISH_FALLBACK_KEYS = frozenset(
    {
        "AI Video Quote Required",
        "AI Video Quote Retained For Retry",
        "AI Video Quote Estimate Incomplete",
        "AI Video Quote Summary",
        "AI Video Quote Summary Singular",
        "AI Video Model",
        "AI Video Model Reference Price",
        "AI Video Model List Load Failed",
        "AI Video Duration Basis Actual",
        "AI Video Duration Basis Estimated",
        "AI Video Material Coverage",
        "AI Video Scene Count",
        "Confirm AI Video Charge",
        "Confirm AI Video Charge Help",
        "Confirm AI Video Charge Required",
        "Custom API Endpoint",
        "API Platform",
        "llm_provider_endpoint_selector.moonshot",
        "llm_provider_endpoint_selector_help.moonshot",
        "llm_provider_endpoint.moonshot.china",
        "llm_provider_endpoint.moonshot.global",
        "llm_provider_authentication_error.moonshot",
        "Local LLM Script Generation",
        "llm_provider_label.apimart",
        "llm_provider_label.openrouter",
        "llm_provider_label.shengsuanyun",
        "LoomLoom Poll Retry Pending",
        "LoomLoom Poll Retry Warning",
        "Resume LoomLoom Status Check",
        "Refresh AI Video Models",
        "Retry AI Video Quote",
        "LoomLoom Quote Summary Singular",
        "LoomLoom Video Terms Reuse Help",
        "Metaso MiniMax H3",
        "Metaso MiniMax H3 Help",
        "Metaso MiniMax API Key",
        "Metaso MiniMax API Key Help",
        "Metaso MiniMax Base URL",
        "Metaso MiniMax Resolution",
        "Metaso MiniMax Resolution Help",
        "Metaso MiniMax Invalid Resolution",
        "Select Metaso MiniMax Resolution",
        "Please Enter the Metaso MiniMax API Key",
        "Metaso MiniMax Billing Notice",
        "Metaso MiniMax Billing Notice Uploaded Audio",
        "Metaso MiniMax Billing Notice Without Script",
        "Confirm Metaso MiniMax Charge",
        "Confirm Metaso MiniMax Charge Help",
        "Confirm Metaso MiniMax Charge Required",
        "Script Generation Method",
        "Script Generation Method Help",
        "Shengsuan Cloud AI Video",
        "Shengsuan Cloud AI Video Help",
        "Shengsuan Cloud API Key",
        "Shengsuan Cloud API Key Help",
        "Shengsuan Cloud API Key Link",
        "Shengsuan Cloud API Key Placeholder",
        "Shengsuan Cloud API Key Required",
        "Shengsuan Cloud API Key Reused",
        "Shengsuan Cloud Batch Script Generation",
        "Selected AI Video Model Unavailable",
        "Selected AI Video Ratio Unavailable",
        "Stop Tracking LoomLoom Run",
        "Stop Tracking LoomLoom Run Help",
        "Unavailable AI Video Model",
        # --- Subtitle style presets, animations, and title overlay keys ---
        "Subtitle Style Preset",
        "Subtitle Casing",
        "As Is",
        "UPPERCASE (TikTok Style)",
        "TikTok Viral Yellow",
        "Alex Hormozi Punch",
        "MrBeast Bold",
        "CapCut Dark Pill",
        "Minimalist Clean",
        "Cyber Neon Aqua",
        "Fire & Alert Red",
        "Golden Luxury",
        "Comic Pop Humor",
        "Viral Barbie Pink",
        "Vintage Cinema Warm",
        "Scale Up (Punch)",
        "Smooth Fade",
        "Slide Up",
        "Shake (Impact)",
        "Video Title Settings",
        "Enable Title / Hook Banner",
        "Title Text",
        "Title Text Help",
        "Title Style",
        "TikTok Yellow Badge",
        "Breaking Red Banner",
        "Neon Cyber Glow",
        "Minimalist Bold White",
        "Golden Luxury Card",
        "Comic Bang Punch",
        "Title Position",
        "Intro (First 4 Seconds)",
        "Full Video",
        "Title Duration",
        "Title Animation",
        # --- Gemini TTS keys ---
        "Gemini TTS Model",
        "Models listed by Google's API for this key; different models have different rate limits and quotas.",
        # --- Cross-post publishing keys ---
        # Short UI labels; English fallback reads fine until a translator picks
        # them up, so keep secondary locales free of these keys.
        "Publish",
        "Publish Scheduled",
        "Publish Failed",
        "Test Connection",
        "Test Connection Help",
        "Test Connection Success",
        "Test Connection Failed",
        "Test Connection Not Configured",
        "Cross Post State Pending",
        "Cross Post State Processing",
        "Cross Post State Complete",
        "Cross Post State Failed",
    }
)
FORMAT_PLACEHOLDER_PATTERN = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})")
MARKDOWN_URL_PATTERN = re.compile(r"\[[^\]]+\]\((https?://[^)]+)\)")


class _TrKeyVisitor(ast.NodeVisitor):
    def __init__(self):
        self.keys = set()

    def visit_Call(self, node):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "tr"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            self.keys.add(node.args[0].value)
        self.generic_visit(node)


def _load_translation(locale):
    data = json.loads((I18N_DIR / f"{locale}.json").read_text(encoding="utf-8"))
    return data.get("Translation", {})


def _required_translation_keys(translations):
    """Return keys that secondary locales must maintain; provider long descriptions fall back to English."""
    return {
        key
        for key in translations
        if key not in ENGLISH_FALLBACK_KEYS
        and not key.startswith(PROVIDER_TIPS_PREFIXES)
    }


def _format_placeholders(value):
    """Extract runtime format variables to prevent translations from missing or renaming them."""
    return set(FORMAT_PLACEHOLDER_PATTERN.findall(value))


def _markdown_urls(value):
    """Extract Markdown link targets; translations may change link text but must not break URLs."""
    return set(MARKDOWN_URL_PATTERN.findall(value))


class TestWebuiI18n(unittest.TestCase):
    def test_saved_ui_language_takes_priority_over_browser_locale(self):
        language = utils.resolve_ui_language(
            saved_language="de",
            browser_locale="zh-CN",
            supported_languages=["zh", "en", "de"],
        )

        self.assertEqual(language, "de")

    def test_browser_locale_is_normalized_to_supported_base_language(self):
        self.assertEqual(
            utils.resolve_ui_language("", "zh-CN", ["zh", "en"]),
            "zh",
        )
        self.assertEqual(
            utils.resolve_ui_language(None, "pt_BR", ["en", "pt"]),
            "pt",
        )

    def test_unsupported_browser_locale_falls_back_to_english(self):
        language = utils.resolve_ui_language(
            saved_language="",
            browser_locale="fr-FR",
            supported_languages=["zh", "en"],
        )

        self.assertEqual(language, "en")

    def test_english_locale_covers_static_webui_labels(self):
        tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
        visitor = _TrKeyVisitor()
        visitor.visit(tree)

        en_keys = set(_load_translation("en"))

        self.assertEqual(sorted(visitor.keys - en_keys), [])

    def test_shengsuanyun_provider_tips_keep_registration_and_model_links(self):
        """Partner entry points and model directories are product configuration; prevent accidental deletion of tracking links during copy edits."""
        expected_urls = {
            "https://www.shengsuanyun.com/?from=CH_XUQ4OTSK",
            "https://global.modelmesh.info/model",
        }

        for locale in ("zh", "en"):
            with self.subTest(locale=locale):
                tips = _load_translation(locale)["llm_provider_tips.shengsuanyun"]
                provider = get_llm_provider("shengsuanyun")
                rendered = tips.format(
                    api_key_url=provider.effective_api_key_url(),
                    default_base_url=provider.effective_default_base_url,
                    default_model=provider.default_model,
                )
                self.assertEqual(_markdown_urls(rendered), expected_urls)

    def test_metaso_api_key_label_keeps_mpt_referral_link(self):
        """Metaso API key entry must keep the MPT referral parameter to preserve the sponsor conversion funnel."""
        expected_url = "https://metaso.cn/minimax-h3/?s=MPT"

        for locale in ("zh", "en"):
            with self.subTest(locale=locale):
                label = _load_translation(locale)["Metaso MiniMax API Key"]
                self.assertEqual(_markdown_urls(label), {expected_url})

    def test_secondary_locales_cover_english_locale(self):
        en_translations = _load_translation("en")
        required_en_keys = _required_translation_keys(en_translations)

        for locale in SECONDARY_LOCALES:
            with self.subTest(locale=locale):
                locale_keys = set(_load_translation(locale))
                self.assertEqual(sorted(required_en_keys - locale_keys), [])

    def test_secondary_locales_do_not_duplicate_provider_tips(self):
        # Provider long descriptions are only maintained in Chinese and English;
        # other languages fall back to English at runtime. Prevent duplicating
        # these keys to avoid half-translated content that won't be maintained.
        for locale in SECONDARY_LOCALES:
            with self.subTest(locale=locale):
                locale_keys = set(_load_translation(locale))
                duplicated_keys = sorted(
                    key for key in locale_keys if key.startswith(PROVIDER_TIPS_PREFIXES)
                )
                self.assertEqual(duplicated_keys, [])

    def test_secondary_locales_do_not_duplicate_english_fallback_keys(self):
        for locale in SECONDARY_LOCALES:
            with self.subTest(locale=locale):
                locale_keys = set(_load_translation(locale))
                self.assertEqual(sorted(ENGLISH_FALLBACK_KEYS & locale_keys), [])

    def test_secondary_locales_cover_static_webui_labels(self):
        tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
        visitor = _TrKeyVisitor()
        visitor.visit(tree)

        for locale in SECONDARY_LOCALES:
            with self.subTest(locale=locale):
                locale_keys = set(_load_translation(locale))
                self.assertEqual(
                    sorted(visitor.keys - locale_keys - ENGLISH_FALLBACK_KEYS),
                    [],
                )

    def test_secondary_locales_preserve_format_placeholders(self):
        en_translations = _load_translation("en")

        for locale in SECONDARY_LOCALES:
            locale_translations = _load_translation(locale)
            for key in _required_translation_keys(en_translations):
                with self.subTest(locale=locale, key=key):
                    self.assertEqual(
                        _format_placeholders(locale_translations[key]),
                        _format_placeholders(en_translations[key]),
                    )

    def test_secondary_locales_preserve_markdown_urls(self):
        en_translations = _load_translation("en")

        for locale in SECONDARY_LOCALES:
            locale_translations = _load_translation(locale)
            for key in _required_translation_keys(en_translations):
                with self.subTest(locale=locale, key=key):
                    self.assertEqual(
                        _markdown_urls(locale_translations[key]),
                        _markdown_urls(en_translations[key]),
                    )

    def test_script_language_options_include_russian(self):
        tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
        support_locales = None

        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id == "support_locales"
                for target in node.targets
            ):
                support_locales = ast.literal_eval(node.value)
                break

        self.assertIsNotNone(support_locales)
        self.assertIn("ru-RU", support_locales)
