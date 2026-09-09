# Agent Working Conventions — MoneyPrinterTurbo

This file is read by every AI coding assistant working in this repository
(Claude Code, Codex CLI, ChatGPT, OpenCode, Antigravity, Cursor, etc.).
If you are an LLM making edits to files in this repo, follow these rules.

## 1. English-only comments and human-readable strings in code

This codebase grew up with Chinese-speaking contributors. Roughly 1,400 lines
still contain Han characters in `#` comments, docstrings, and log/error
string literals. New contributors keep adding more in whichever language they
think in. We want the source readable by an English-only reviewer (human or
LLM) without sitting down for a repo-wide translation marathon — so we
clean up incrementally, only the file you are currently touching.

### Rule (applies to every LLM edit in this repo, every time)

When you edit a file in this repo — whether you are Claude, Codex, ChatGPT,
OpenCode, Antigravity, Cursor, or anything else — you must do BOTH of the
following, scoped strictly to the file(s) you are editing:

1. **Translate any existing Asian-language text in the touched file** —
   find every `#` line comment, every docstring (`"""..."""` /
   `'''...'''`), and every user-facing string literal (log messages,
   error messages, `raise SomeError("...")`, `HTTPException` detail,
   `print(...)` arguments) that contains Asian-language characters, and
   rewrite the human-readable text into English. Preserve identifiers,
   imports, decorators, format specifiers, and inline code references.

2. **Write any new comments and strings in English** — do not introduce
   new Han/Hiragana/Katakana/Hangul characters in code you write.

### Scope: touched files ONLY

Do not run a repo-wide sweep. Limit translation to:
- The file you are currently editing.
- Any file you re-read and re-edit as part of the same task.
- For a new file you are creating, the entire new file.

If you are tempted to "while I'm here" translate another file, do not.
The point is incremental cleanup driven by real edits.

### What counts as "Asian-language text"

Detect with these Unicode ranges — any one character is enough to flag
the line:

| Script                  | Range              |
|-------------------------|--------------------|
| CJK Unified Ideographs  | U+4E00–U+9FFF      |
| CJK Extension A         | U+3400–U+4DBF      |
| Hiragana                | U+3040–U+309F      |
| Katakana                | U+30A0–U+30FF      |
| Katakana Phonetic Ext.  | U+31F0–U+31FF      |
| Hangul Syllables        | U+AC00–U+D7AF      |
| Hangul Jamo             | U+1100–U+11FF      |
| CJK Compat. / Fullwidth | U+3300–U+33FF, U+FF00–U+FFEF |

### What NOT to translate

- Code identifiers, keywords, attribute names, import paths — never.
- `webui/i18n/*.json` — those are the localization source of truth.
- Translation-key constants that reference i18n key names.
- Regex patterns, SQL strings, shell commands, file paths, URLs, base64
  blobs, hex literals, UUIDs — even if they contain Asian-looking bytes,
  leave the bytes alone.
- Test assertions that pin exact Chinese text (search the test first).
- `README*.md`, `docs/**/*.md`, `LICENSE` — unless the user asks.
- Already-English comments — leave them.
- Full-width punctuation alone in otherwise-English text — leave it.

### Translation style

- Literal enough that a code reviewer trusts the diff.
- Preserve parenthetical asides, examples in backticks, code references.
- Match the project's voice: short, declarative, lowercase (`# keep this
  thread-safe`, `# lazy init on first use`).
- Do not editorialize. Do not delete the comment. Do not add new
  commentary.
- If a term is genuinely ambiguous, keep the original term and add a
  short English gloss in parentheses once, rather than mistranslating.

### Verification

Before you finish:
- Re-read the file.
- Sweep for any remaining Asian-script characters in comments / docstrings
  / strings. Zero is the target. If any remain and you cannot translate
  them confidently, name them in your reply so the human can decide.

## 2. Other project conventions

- Python 3.10+, FastAPI, loguru, threading + bounded executors. Match
  the surrounding style — `from __future__ import annotations`, type hints,
  frozen dataclasses, `from X import Y` form.
- Tests live under `test/`, runnable with `pytest`. Smallest test that
  proves the fix.
- Don't add a new dependency when stdlib or an already-installed one
  covers it.
- Don't refactor unrelated code in the same edit — keep the diff to the
  smallest working change.