
# AGENTS.md

This file defines strict operating rules for coding agents working in this repository.

## 1. Required Reading Order (Before Any Code Change)
1. Read `IMPLEMENTATION_PLAN.md`.
2. Read `SOLUTION_DESIGN.md`.

You must follow both documents in this order:
- `IMPLEMENTATION_PLAN.md` defines project scope and target architecture.
- `SOLUTION_DESIGN.md` defines what is actually implemented right now.

If they differ, treat `SOLUTION_DESIGN.md` as the source of truth for current code behavior, and `IMPLEMENTATION_PLAN.md` as the boundary for what may be added.

## 2. Scope Guardrails
- Do not implement features outside `IMPLEMENTATION_PLAN.md`.
- Do not add "nice-to-have" extras, speculative abstractions, or optional systems not explicitly needed.
- Keep work aligned with the current 3-layer plan and current implementation status.

## 3. Architecture and Structure Rules
- Respect existing package boundaries under `src/` (`training`, `orchestration`, `evaluation`, `utils`).
- Place new code in the correct layer. Do not mix responsibilities across layers.
- Do not move or rename modules unless the task explicitly requires it.
- Keep APIs minimal and explicit; prefer simple data flow over hidden coupling.
- For larger design or feature work, use a modular design mindset instead of long monolithic scripts.
- Prefer thin orchestration scripts that call reusable module functions/classes.
- Module APIs must be explicit: define expected inputs and promised outputs clearly.

## 4. Coding Style Rules
- Keep code simple, short, and readable.
- Avoid long files, large classes, and convoluted functions.
- Avoid overly verbose code
- Avoid over-engineering and unnecessary indirection.
- Keep comments minimal and meaningful.
- No `try/except` around imports.
- Never modify core package behavior just to make a test or script pass.
- Always add concise docstrings to helper functions, including what they return.

## 5. Change Discipline
- Make the smallest safe change that solves the task fully.
- Preserve existing behavior unless behavior change is explicitly required.
- Do not perform broad refactors unless requested.
- Keep naming consistent with existing code and config conventions.

## 6. Validation
- Run focused checks relevant to your change (tests, lint, or targeted script runs).
- Prefer quick, local validation first, then broader validation when needed.
- If something cannot be validated, state that clearly.
- NEVER modify functionality of core functions or APIs to perform a test or run a test script

## 7. Documentation Update Policy
Update `SOLUTION_DESIGN.md` whenever implementation changes meaningfully, including:
- new modules/components,
- changed public APIs,
- behavior or data-flow changes,
- config or runtime contract changes.

Do not update `SOLUTION_DESIGN.md` if you only did bug fixing, small minor changes, or updates to the documentation (not core source functionality).

For multi-step implementation work, revisit and refresh `SOLUTION_DESIGN.md` periodically so it stays accurate.

Do NOT change README.md

## 8. Environment and Command Conventions
- Use `uv` for environment and execution.
- Typical commands:
	- `uv sync`
	- `uv run ...`

## 9. Branching Strategy
- Do not develop directly on `main`.
- Always create and use a feature branch for implementation work.
- Keep feature branches short-lived and light.
- Scope each feature branch to a single functionality whenever possible.

## 10. Priority Order for Agent Decisions
When rules conflict, follow this priority:
1. Direct user request.
2. `IMPLEMENTATION_PLAN.md` scope constraints.
3. `SOLUTION_DESIGN.md` current-state constraints.
4. This `AGENTS.md` style and process rules.