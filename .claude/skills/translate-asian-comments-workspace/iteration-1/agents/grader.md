# Grader prompt template

You are grading a run for the `translate-asian-comments` skill.

## Inputs
- EVAL_DIR: /Users/dani/Developer/Dev/MoneyPrinterTurbo/.claude/skills/translate-asian-comments-workspace/iteration-1/eval-<N>-<name>/<configuration>/
- Output files in EVAL_DIR/outputs/:
  - *.final — the resulting Python source file
  - changes.diff — git diff vs the original
  - notes.md — what the agent did
  - metrics.json — tool-call counts and timing

## Assertions
Read EVAL_DIR/eval_metadata.json — its `assertions` list is what you must check.

## How to grade

For each assertion in eval_metadata.json:
1. Determine whether the assertion is verifiable from the output files. If yes, run the appropriate check.
2. Use Bash + grep + python to programmatically verify. For each assertion produce:
   - `text`: the assertion string verbatim from eval_metadata.json
   - `passed`: true | false
   - `evidence`: a short string quoting what you saw (file:line, grep match, command output, etc.)

### Programmatic checks you should run

- For "X is in English / no Han characters": run
  ```python
  python -c "import re,sys; pat=re.compile(r'[一-鿿㐀-䶿]'); print('HITS' if pat.search(open(sys.argv[1]).read()) else 'CLEAN')" <file>
  ```
- For "Only <file> is modified": run `git -C /Users/dani/Developer/Dev/MoneyPrinterTurbo status --porcelain` and verify no other paths appear.
- For "Function X added": `grep -n "def get_active_task_count" <file>`.
- For "Logic unchanged": spot-check that key code paths are intact.

Save the result as grading.json in EVAL_DIR/, with shape:
```json
{
  "expectations": [
    {"text": "...", "passed": true, "evidence": "..."},
    ...
  ],
  "summary": {"passed": N, "failed": M, "total": N+M, "pass_rate": passed/(passed+failed)}
}
```

Important: do not modify any source file. Read-only grading.