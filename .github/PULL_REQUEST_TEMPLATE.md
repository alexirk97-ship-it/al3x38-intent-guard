## Summary

Describe the changes made in this Pull Request and what issue/use case it addresses.

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature / Embedder adapter
- [ ] Vector math / Threshold / `ConscienceCompass` improvement
- [ ] Documentation / Examples / Tests

## Core Invariants Checklist

Before submitting, please verify:

- [ ] **Zero Heavy Dependencies:** No new third-party dependencies added to core (`numpy` + stdlib only).
- [ ] **Async-Safe:** All I/O and embedding calls remain non-blocking (`asyncio` / `asyncio.to_thread`).
- [ ] **Tested:** Existing tests pass and new behavior has been verified locally.
- [ ] **Docs Updated:** `README.md` / `README_RU.md` updated if public config fields changed.