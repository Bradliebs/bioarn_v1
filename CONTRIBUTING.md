# Contributing

Thanks for contributing to Bio-ARN 2.0.

## Development environment

```powershell
git clone <your-fork-or-clone-url>
cd bioarn
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch
pip install -e .[dev]
```

Run the test suite before and after changes:

```powershell
pytest
```

## Code style

- Use **type hints** on public functions, methods, and dataclasses.
- Add **docstrings** for public classes and non-obvious behavior.
- Keep changes **surgical**: update only the modules, tests, and docs needed for the change.
- Prefer **local learning rules** over backprop in the core architecture.
- If you introduce a backprop-dependent experiment, isolate it clearly and document why it exists.
- Preserve the repository’s current style: small functions, dataclass outputs, and explicit tensor shapes.

## Testing requirements

- All new behavior needs tests.
- Extend an existing test module when possible; create a new one only when the scope is clearly separate.
- Add unit tests for component-level changes and an integration test when the perception-action loop changes.
- For benchmark or energy changes, update the script and document how the result should be reproduced.

## Pull request process

1. Create a focused branch.
2. Implement the change and update docs if behavior changes.
3. Run `pytest`.
4. Summarize the motivation, approach, and verification in the PR description.
5. Call out any benchmark, energy, or API-surface impact explicitly.

## Adding new components

When adding a new encoder, learning rule, or backend:

1. Place it in the relevant package (`sensorimotor`, `reward`, `predictive`, `hardware`, etc.).
2. Export it from that package’s `__init__.py` if it is part of the public API.
3. Add config surface area only when the setting is needed for reproducibility or tuning.
4. Add API documentation in `docs/api_reference.md` and architecture notes in `docs/architecture.md` if the component changes system flow.
