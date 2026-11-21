# Contributing to `al3x38-intent-guard`

Thank you for your interest in contributing, testing edge cases, or forking **`al3x38-intent-guard`**!

Whether you want to tune vector thresholds, add new embedder adapters, write benchmarks, or port the core math to another language, contributions and forks are warmly welcome under the MIT License.

---

## Core Architectural Rules (Invariants)

To keep the kernel fast, predictable, and easy to run on modest CPU/edge hardware, please follow these principles when modifying `src/al3x38_intent_guard/`:

1. **Zero Heavy Core Dependencies:** The core engine (`core.py`, `conscience_compass.py`, `parsers.py`) must rely **only** on the Python standard library and `numpy`. Optional heavy libraries belong strictly in `[project.optional-dependencies]`.
2. **Invariants Over Static Rules:** Do not add hardcoded keyword blacklists or domain-specific `if/else` patches to the kernel. Security checks should be expressed through vector geometry, entropy, or configurable regex/tool hooks.
3. **Async-First & Non-Blocking:** All I/O operations (network embedder calls, ONNX inference, disk state persistence, audit logging) must remain non-blocking via `asyncio` / `asyncio.to_thread`.
4. **Language Agnosticism:** Keep differential semantic anchors and drift metrics compatible with multilingual and English embedding models.

---

## Local Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/alexirk97-ship-it/al3x38-intent-guard.git
   cd al3x38-intent-guard
   ```

2. Create a virtual environment and install in editable mode:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # Linux/macOS: source .venv/bin/activate
   pip install -e ".[offline]"
   pip install pytest pytest-asyncio
   ```

3. Run tests (if available in `tests/`):
   ```bash
   pytest
   ```

---

## Submitting Pull Requests

* Keep PRs focused on a single feature, bug fix, or threshold/math improvement.
* If you propose changes to `ConscienceCompass` (`D * L -> S`) or `GuardConfig` defaults, include a brief before/after example showing how cosine drift or coherence scores behave on sample prompts.
* Update `README.md` / `README_RU.md` if you add new public configuration options.