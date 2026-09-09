Translated all Chinese (# comments, docstrings) in app/services/task.py to English.
No logic was changed. Excluded the test data string "金钱的作用" in the `if __name__ == "__main__"` block — the skill scopes excludes test fixtures that pin exact text.
No CJK characters remain in any comment, docstring, or log/error string literal.
