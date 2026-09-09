---
name: translate-asian-comments
description: Translate any Asian-language text (Chinese, Japanese, Korean — Han/Hiragana/Katakana/Hangul) found in code comments, docstrings, and user-facing string literals (log messages, error messages) into English — but ONLY for files the LLM is currently touching. The cross-LLM source of truth for this rule lives in /AGENTS.md (read by Claude Code, Codex, ChatGPT, OpenCode, Antigravity, Cursor, etc.). This skill is the Claude-Code-native implementation: it triggers proactively whenever a Read/Edit/Write tool call touches a file in the MoneyPrinterTurbo repo, applies the AGENTS.md rule, and verifies the result. Use this skill before AND after editing any .py, .js, .ts, .tsx, .jsx, .vue, .go, .java, .kt, .swift, .rs, .css, .html, .yaml, .toml, .json, .sh file in this repo — regardless of which LLM is doing the edit. Do not wait to be asked. Never translate strings inside i18n JSON bundles (webui/i18n/*.json). Never translate string content that is part of a regex, SQL query, shell command, file path, env-var name, import path, URL, base64 blob, or test fixture that asserts on exact text.
---

# Translate Asian-Language Comments To English (Touched Files Only)

## Why this skill exists

MoneyPrinterTurbo grew up with Chinese-speaking contributors. Roughly 1,400 lines in `app/`, `cli.py`, and `main.py` still contain Han characters in `#` line comments, docstrings, and log/error string literals. New contributors keep adding more in whichever language they think in. The user wants the codebase readable by an English-only LLM (or human) without sitting down for a repo-wide translation marathon — so we clean up incrementally, only the file the LLM is currently touching.

The full cross-LLM rule (for every AI coding tool that reads AGENTS.md) lives at `../AGENTS.md` relative to this skill. This SKILL.md is the Claude-Code-native deep reference: read it for the exact detection regex, the verification checklist, and the exclusion list. For the why and the rule itself, defer to AGENTS.md.

## When this skill triggers

This skill fires for every LLM edit in this repo. Concrete signals:

- Claude is about to call `Edit` or `Write` on any file under `/Users/dani/Developer/Dev/MoneyPrinterTurbo/` other than the explicit exclusion paths below.
- A user pastes a file with Asian comments and asks for any change to it.
- A user asks "translate the comments in `<file>`" — even outside an edit cycle.
- Another LLM (Codex, ChatGPT, OpenCode, Cursor, Antigravity) is being driven through Claude Code and its output is about to be applied as a file edit.

If an Edit/Write tool call is happening and the target file is inside this repo, this skill must run. Do not skip because "the change is small" — the cleanup is also small.

## What counts as "Asian-language text" in scope

Detect with these Unicode ranges (any one character is enough to flag the line):

| Script | Range | Notes |
|---|---|---|
| CJK Unified Ideographs | U+4E00–U+9FFF | Simplified + Traditional Chinese, Kanji |
| CJK Extension A | U+3400–U+4DBF | Rare CJK |
| Hiragana | U+3040–U+309F | Japanese |
| Katakana | U+30A0–U+30FF | Japanese |
| Katakana Phonetic Ext. | U+31F0–U+31FF | Japanese |
| Hangul Syllables | U+AC00–U+D7AF | Korean |
| Hangul Jamo | U+1100–U+11FF | Korean |
| CJK Compatibility / Fullwidth | U+3300–U+33FF, U+FF00–U+FFEF | Punctuation + symbols |

If a line contains any character from these ranges, the line is in scope.

## Scope: touched files ONLY

The user's hard requirement: do **not** run a repo-wide sweep. Limit translation to:

- The file the LLM is currently editing (the path passed to `Edit`/`Write`).
- Any file the LLM explicitly reads as part of the same task and then re-edits.
- For new file creation (`Write` of a brand-new path), the entire new file is in scope.

If you are tempted to "while I'm here" translate another file, do not. The point is incremental cleanup driven by real edits.

## What to translate

In every in-scope file, walk the file and translate any in-scope line so the **comment / docstring / string content** becomes English. Code identifiers, keywords, imports, decorators, and type annotations stay as-is — even if they happen to be transliterated CJK words (e.g. `视频` is never a Python identifier in this repo, but if it were, leave it).

Three categories to rewrite:

1. **Line comments** — every `#` that starts a comment segment. If only part of the line after `#` is Asian, translate that part.
2. **Docstrings** — the literal text between `"""..."""` or `'''...'''` immediately following a module / class / function / method definition. Keep the quotes, keep the indent, translate the inner text.
3. **User-facing string literals** — log messages, `logger.info(...)` / `logger.warning(...)` / `logger.error(...)` arguments, `raise SomeError("...")` messages, returned error strings, FastAPI `HTTPException` detail strings, and `print(...)` arguments. Translate the human-readable text only.

## What NOT to translate

- **Code identifiers, keywords, attribute names, import paths** — never.
- **i18n JSON bundles** under `webui/i18n/*.json` — those are the translation source of truth; the values must stay in their target language.
- **Translation-key constants** that reference i18n key names (e.g. `_("video.task_failed")`).
- **Regex patterns, SQL strings, shell command strings, file paths, URLs, base64 blobs, hex literals, UUIDs** — even if they contain Asian-looking bytes, leave the bytes alone.
- **Test assertions that pin exact text** — if a test asserts `"expected: 中文"` to verify i18n output, leave it. (Search the test before translating any string.)
- **`README*.md`, `docs/**/*.md`, `LICENSE`** — the user wants English in code, not in the existing localized docs. Don't touch them unless the user explicitly asks.
- **Comments that are already entirely English** — leave them.
- **Comments that are entirely blank or pure whitespace** — leave them.

## How to translate

1. Identify the in-scope file(s) — the path from the current `Edit`/`Write` call, plus any file re-read and re-edited in the same task.
2. Read the file(s) with `Read` to get exact byte content.
3. For each file, scan every line. For each in-scope line:
   - Extract the comment / docstring / string content.
   - Translate the natural-language portion to clear, idiomatic English. Preserve technical terms (function names, class names, config keys) verbatim. Preserve any inline code, backticks, `{}`-style placeholders, percent-format specifiers (`%s`, `%(name)s`), and f-string interpolation braces.
   - Keep the same indentation and quote style.
4. Apply the changes with `Edit` (preferred — preserves surrounding context) or `Write` (for new files or rewrites too tangled for surgical edits).
5. After editing, re-read the file and run a regex sweep to confirm no in-scope Asian characters remain. If any do, fix them before declaring done.
6. If a translation is genuinely ambiguous (a domain-specific term, a contributor nickname, a project-internal phrase), keep the original term and add a short English gloss in parentheses once, rather than mistranslating.

### Detection regex

```python
import re
ASIAN = re.compile(
    r"[一-鿿㐀-䶿぀-ゟ゠-ヿ㌀-ヿ가-힯ᄀ-ᇟ＀-￯]"
)
```

One character is enough to flag a line. Do not flag based on full-width punctuation alone if the surrounding text is English — that is intentional stylistic choice and should stay.

### Translation style

- Be literal enough that a code reviewer can trust the diff. The reader is going to look at the line in a code review.
- Keep parenthetical asides, examples in backticks, and code references intact.
- Match the project's voice: short, declarative, lowercase to match the existing English comments (`# keep this thread-safe`, `# lazy init on first use`).
- Do not editorialize. Do not add commentary. Do not remove the comment.

### Example diffs (these are the kinds of swaps the skill produces)

```diff
-# 视频生成和跨平台发布都可能继续读取任务目录。统一视为忙碌状态，
-# 可以避免 API 与 WebUI 分别维护规则后出现一个允许删除、另一个禁止
-# 删除的不一致行为。
+# Both video generation and cross-platform publishing can keep reading
+# the task directory. Treating both as busy avoids API/WebUI disagreeing
+# on whether a delete is allowed.
```

```diff
-def is_task_busy(task: dict | None) -> bool:
-    """判断任务是否仍在生成或发布，供所有删除入口复用。"""
+def is_task_busy(task: dict | None) -> bool:
+    """Return True while the task is still generating or publishing; shared by every delete entry point."""
```

```diff
-            raise SoniloError(f"Sonilo 调用失败: {exc}") from exc
+            raise SoniloError(f"Sonilo call failed: {exc}") from exc
```

## Verification before declaring done

- [ ] Re-read the modified file.
- [ ] Run the Asian-character regex over the file and confirm zero matches.
- [ ] Confirm no test file was modified unless the test pinned Chinese text (in which case leave it and tell the user).
- [ ] If a string in `webui/i18n/*.json` was accidentally touched, revert it.
- [ ] If a translation could not be done confidently, leave the original line and flag it in the final reply so the user can decide.

## Output

State plainly:

- File(s) touched and the count of lines translated.
- Anything deliberately skipped (test assertions, i18n bundles, ambiguous terms) with the file:line of each.
- Any Asian script characters that remain in the file (there should be none — if there are, name them).

No essays. No "I noticed you also have Chinese in X file" drive-bys. Translation outside the touched files is out of scope.