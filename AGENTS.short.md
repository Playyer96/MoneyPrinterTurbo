# Short paste version — for ChatGPT / Claude Desktop Custom Instructions

This file is a one-shot, ~200-token summary of the rule. Paste it into:

- **ChatGPT** → Settings → Personalization → Custom Instructions ("How
  would you like ChatGPT to respond?"). Applies to every conversation
  on this account, no project setup needed.
- **Claude Desktop** → Settings → Profile → Custom Preferences (or
  paste into a Project's "Project instructions"). Applies to every
  Claude Desktop conversation, or scoped to one Project.

It is intentionally short. The full rule with detection ranges,
exclusions, and verification lives at `AGENTS.md` in the repo root and
is read automatically by Claude Code, Cursor, Codex CLI, OpenCode,
Antigravity, Continue, Cody, Windsurf, aider, etc.

---

Working on the MoneyPrinterTurbo Python repo. Apply this rule on every edit:

1. Translate any existing Asian-language (# comments, docstrings, log/error
   strings) in the touched file to English. Skip `webui/i18n/*.json`,
   regex/SQL/shell/paths/URLs, test fixtures, and base64.
2. New comments and strings you add must be English.
3. Scope is touched files only — never run a repo-wide sweep.

Style: short, declarative, lowercase. Preserve backticks, format
specifiers, and code references. If a term is ambiguous, gloss it in
parentheses; do not mistranslate.