---
name: anti-loop-harness
description: Universal termination protocol constraint. Forces explicit task end patterns.
---

# Instructions
1. When you have completed the user's primary prompt, immediately append the termination string.
2. DO NOT output code changes, task lists, or recursive summaries if the edit or answer is fully fleshed out.
3. If no further files need reading, immediately yield a clear text statement ending with the string: <promise>DONE</promise>
