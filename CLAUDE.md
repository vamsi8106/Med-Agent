@AGENTS.md

## Claude Code specific
- Use plan mode for changes touching more than 3 files.
- Always run `make pre-commit` before considering a task done.
- Never pre-create empty packages — create modules at their phase.
- When adding a dependency: `uv add <pkg>` for runtime, `uv add --group dev <pkg>` for dev.
- Every new file must have type annotations on all function signatures including `-> None`.
- When creating a new module, also create its corresponding test file in `tests/unit/`.
- Commit messages: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:` prefix.
- If unsure about a medical term or clinical logic, ask — do not guess.
- Patient data in tests and demos must be clearly synthetic (use names like "Patient Alpha", IDs like "P-TEST-001").
