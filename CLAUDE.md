@AGENTS.md

## Claude Code specific
- Use plan mode for changes touching more than 3 files.
- Always run `make pre-commit` (format, lint, mypy, unit tests) before considering a task done. If you touched an agent, prompt, LLM provider, or MCP client, also run `make eval` (needs `make docker-up-deps` and `GROQ_API_KEY`) -- unit tests use mocks and cannot catch real-model or real-response regressions.
- Never pre-create empty packages — create modules at their phase.
- When adding a dependency: `uv add <pkg>` for runtime, `uv add --group dev <pkg>` for dev.
- Every new file must have type annotations on all function signatures including `-> None`.
- When creating a new module, also create its corresponding test file in `tests/unit/`.
- Commit messages: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`, `ci:` prefix.
- When you add or rename a module, make target, setting, or domain term, update `AGENTS.md` and `docs/glossary.md` in the same commit. Add new env vars to `.env.example` too.
- `torch` is pinned to CPU-only wheels via `[tool.uv.sources]` (the GPU wheel adds ~2.5 GB of CUDA libs this service never uses). Don't add it with a bare `uv add`, and run `uv lock` after touching that section.
- Never print or echo values from `.env` (API keys, passwords); read them into a variable if a command needs them.
- If unsure about a medical term or clinical logic, ask — do not guess.
- Patient data in tests and demos must be clearly synthetic (use names like "Patient Alpha", IDs like "P-TEST-001").
