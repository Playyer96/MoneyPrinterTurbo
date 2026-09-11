---
name: directory-sandbox
description: Guardrail strict security constraint. Enforces that all file read, write, create, and bash operations happen ONLY within the current working directory.
---

# Instructions
1. You are strictly sandboxed. You are forbidden from reading, writing, updating, or creating files outside of the current working directory.
2. Under no circumstances should you navigate up the directory tree using `../` or access global system roots (such as `C:\`, `D:\`, `%USERPROFILE%`, etc.).
3. If an automated script or a tool request attempts to output a file modification outside of the current folder path, you must immediately abort the task and return a security exception message to the user.
4. All relative paths must resolve strictly within the local workspace root (`.`).
