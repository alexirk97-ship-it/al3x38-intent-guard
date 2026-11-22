**English** | [Русский](README_RU.md)

# al3x38-intent-guard

[![PyPI version](https://img.shields.io/pypi/v/al3x38-intent-guard.svg)](https://pypi.org/project/al3x38-intent-guard/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/personalized-badge/al3x38-intent-guard?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/al3x38-intent-guard)
[![Dependencies](https://img.shields.io/badge/dependencies-pure%20NumPy-orange.svg)](#installation)
[![Async](https://img.shields.io/badge/async-asyncio%20ready-blueviolet.svg)](#quick-start)
[![GitHub stars](https://img.shields.io/github/stars/alexirk97-ship-it/al3x38-intent-guard?style=flat&color=yellow)](https://github.com/alexirk97-ship-it/al3x38-intent-guard/stargazers)
[![Tests](https://github.com/alexirk97-ship-it/al3x38-intent-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/alexirk97-ship-it/al3x38-intent-guard/actions/workflows/ci.yml)

**`al3x38-intent-guard`** is an asynchronous behavioral guardrail and vector coherence compass for LLM agents in Python. Instead of relying on brittle keyword blacklists, it verifies **deep semantic alignment** between the user's original intent, the session's long-term goal, the model's response, and privileged system actions (`Function Calling`).

---

## Philosophy & Design Principle

Most AI safety tools are built like endless walls of prohibitions: static regex filters, word blacklists, and heavy external classifier models. In living systems, however, stability does not come from memorizing millions of prohibitions—it comes from holding a **reference invariant** (an internal compass) and sensing deviation from harmony in real time.

At the core of `al3x38-intent-guard` lies a strict, geometric model of behavioral coherence:
* **Truth without respect (D > 0, L < 0)** turns into cold, destructive hostility.
* **Politeness without truth (D < 0, L > 0)** becomes hidden manipulation and camouflage ("polite prompt injection," where an agent smiles at the user while silently attempting to delete files).
* **True alignment (Coherence, S = √(D · L))** exists only when **Transparency/Truth (D)** and **Constructive Respect (L)** are simultaneously positive and aligned with the session's **Reference Goal**.

The kernel translates this principle into fast, dependency-free vector math on `numpy`, running in milliseconds on standard CPUs without expensive GPU clusters.

---

## Defense Pipeline (v1.2.0)

```text
[User Prompt + LLM Response + Tool Calls + Logits]
                        │
                        ▼
        0. Session Trust & Time Recovery
           └─► If trust <= suspend_limit ──► [SUSPENDED]
                        │
                        ▼
        1. Secret & System Prompt Leak Scanner
           └─► API keys / prompt injection leak ──► [BLOCK] (-trust)
                        │
                        ▼
        2. Semantic Cross-Validation (Embeddings)
           ├─► Prompt <-> Response drift
           └─► Prompt <-> Restricted Tool Arguments drift
                        │
                        ▼
        3. Logit Entropy & Gray-Zone Arbitrator
           └─► High drift or high entropy ──► [BLOCK] (-non-linear penalty)
                        │
                        ▼
        4. Vector Compass (Session Goal Check)
           ├─► drift > max_goal_drift ──────────► [BLOCK] (-non-linear penalty)
           └─► drift > goal_exploration_radius ─► [COURSE_CORRECTION]
                        │
                        ▼
        5. Conscience Compass (Differential Axes D ⋅ L → S)
           ├─► S < 0 & gradient > +0.08 ────────► [COURSE_CORRECTION] (redemption)
           └─► S < 0 & degrading trajectory ────► [BLOCK] (-non-linear penalty)
                        │
                        ▼
        6. Low-Trust Quarantine Check
           └─► Restricted tool & trust < min_trust ──► [NEED_APPROVAL]
                        │
                        ▼
               [ALLOW] (+trust_step_up)
```

---

## Installation

Minimal installation (requires only `numpy`):
```bash
pip install al3x38-intent-guard
```

With built-in offline CPU embeddings (no Ollama, internet, or external API required):
```bash
pip install al3x38-intent-guard[offline]
```

---

## Quick Start

### 1. Local Ollama, Vector Compass & Tool Verification
```python
import asyncio
from al3x38_intent_guard import (
    ActionVerdict,
    IntentGuardCore,
    OllamaLocalEmbedder,
    parse_ollama_payload,
)

async def main():
    embedder = OllamaLocalEmbedder(host="http://localhost:11434", model="bge-m3")
    guard = IntentGuardCore(embedder)

    # Set a Vector Compass anchor for the session
    await guard.set_session_goal("user_42", "Customer order status and shipping support")

    # Raw payload from Ollama (/api/chat) containing a hidden rm_dir injection
    raw_response = {
        "message": {
            "content": "Sure, checking your order status right now.",
            "tool_calls": [
                {"function": {"name": "rm_dir", "arguments": {"path": "/var/www"}}}
            ]
        }
    }

    telemetry = parse_ollama_payload(
        session_id="user_42",
        prompt="Where is my order #105?",
        payload=raw_response
    )

    verdict, reason = await guard.check(telemetry)
    if verdict != ActionVerdict.ALLOW:
        print(f"Execution halted [{verdict.value}]: {reason}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Direct Coherence Evaluation (`D`, `L`, `S` Axes) on CPU
```python
import asyncio
from al3x38_intent_guard import IntentGuardCore, OfflineFastEmbedder

async def main():
    embedder = OfflineFastEmbedder(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    guard = IntentGuardCore(embedder)

    state = await guard.get_conscience_state(
        session_id="user_42",
        text="Let's honestly review the bug in the code and fix it together calmly."
    )
    print(f"D={state.friendship_coh}, L={state.love_coh} -> S={state.conscience_score} [{state.verdict}]")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. OpenAI / LM Studio / vLLM Endpoints
```python
from al3x38_intent_guard import IntentGuardCore, OpenAICompatEmbedder

embedder = OpenAICompatEmbedder(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="nomic-embed-text"
)
guard = IntentGuardCore(embedder)
```

### 4. Auto-Calibrate Thresholds for Any Embedding Model
```python
import asyncio
from al3x38_intent_guard import IntentGuardCore, OllamaLocalEmbedder

async def main():
    embedder = OllamaLocalEmbedder(host="http://localhost:11434", model="bge-m3")

    # Automatically compute optimal thresholds for your embedding space
    report = await IntentGuardCore.auto_calibrate(embedder)
    print(f"Calibrated min_harmony={report.recommended_min_harmony}, destructive={report.recommended_destructive_threshold}")

    # Initialize the guard with the calibrated configuration
    guard = IntentGuardCore(embedder, cfg=report.config)

if __name__ == "__main__":
    asyncio.run(main())
```

### 5. High-Load SQL Persistence & Agent Self-Correction (`v1.2.0`)
```python
import asyncio
from al3x38_intent_guard import IntentGuardCore, OllamaLocalEmbedder, SQLStateStore

async def main():
    store = SQLStateStore.from_sqlite("cluster_guard.db")
    guard = IntentGuardCore(OllamaLocalEmbedder(model="bge-m3"), state_store=store)

    draft = await guard.self_reflect_draft("agent_alpha", "Maybe later, it depends on many things...")
    if draft and draft.needs_self_correction:
        print("Revise draft first:", draft.steering_hint)

    guide = await guard.analyze_perspective("user_42", "Everything is terrible, I don't want to check any facts!")
    print("Dominant:", guide.dominant_axis, "| Blind spot:", guide.blind_spot_axis)
    print("LLM hint:", guide.steering_hint)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## What's New in v1.2.0

* **4D Cognitive Decision Matrix & Mutual Agent Teaching (`CognitiveMatrixEngine`):** Projects utterances onto 4 orthogonal poles (`D`, `L`, `R`, `O`), maintains a compact per-entity EMA profile (`~250 bytes`), detects blind spots, and provides `self_reflect_draft()` + `analyze_perspective()` so AI agents self-correct before answering and mutually calibrate each other.
* **High-Load Sharded Concurrency (`ShardedLockPool`):** Uses a `256`-shard loop-safe lock pool so thousands of users run in parallel without blocking each other.
* **Pluggable Persistence (`StateStore` Protocol: JSON, SQLite, MySQL, PostgreSQL):** Adds `JsonFileStateStore` (with debounce flush) and `SQLStateStore` (DB-API 2.0 adapter for SQLite, MySQL, and PostgreSQL) plus `export_profile()` / `import_profile()`.

## What's New in v1.1.5

* **Embedding Auto-Calibration (`CompassCalibrator`):** Added `await IntentGuardCore.auto_calibrate(embedder)` to automatically compute optimal `ConscienceCompass` thresholds tailored to the cosine geometry of your specific embedding model (`bge-m3`, `nomic-embed-text`, `MiniLM`).
* **Dual-Contour Anti-Flood SHA-256 Cache:** `LRUEmbedderCache` splits memory into a clean cache (`512` slots) and an isolated threat quarantine cache (`128` slots) indexed by `SHA-256`. Repeated spam/DDoS attacks hit the quarantine cache in $O(1)$ with zero CPU/API overhead while never evicting legitimate dialogue vectors.

## What's New in v1.1.4

* **Bilingual Semantic Anchors (`EN + RU`):** `ConscienceCompass` now uses bilingual reference anchors by default (`compass_lang="bilingual" | "en" | "ru"`), ensuring accurate coherence projections across both English-only (`nomic-embed-text`) and multilingual (`bge-m3`, `MiniLM`) embedding models.
* **Explicit Polite Deception Verdict (`POLITE_DECEPTION`):** Separated high-politeness manipulation (`D < 0, L > 0`) from general neutral imbalance so disguised social engineering is explicitly flagged and blocked before privileged tools execute.

## What's New in v1.1.3

* **Conscience Compass (`ConscienceCompass` & `ConscienceState`):** Introduced differential semantic axes measuring **Truth/Directness ($D$)** and **Constructive Respect ($L$)**. By subtracting anti-anchor vectors from reference anchors (V+ − V−), the kernel strips away shared linguistic background noise and exposes "polite deception" or hidden manipulation (S = √(D · L)).
* **Trajectory Gradient & Redemption ($\Delta S / \Delta t$):** Tracks a sliding window of recent steps per session. If a user or agent stumbles into negative coherence ($S < 0$) but turns sharply toward honest, constructive dialogue on the next turn ($\Delta S > 0.08$), hard blocking yields to a soft `COURSE_CORRECTION`.
* **Diagnostic Inspection (`get_conscience_state`):** Exposes real-time projections (`friendship_coh`, `love_coh`, `conscience_score`, `gradient`) for telemetry, HUD overlays, and audit analytics.

---

## Documentation & Links

* **GitHub:** [alexirk97-ship-it/al3x38-intent-guard](https://github.com/alexirk97-ship-it/al3x38-intent-guard)
* **PyPI:** [al3x38-intent-guard](https://pypi.org/project/al3x38-intent-guard/)
* **Raspberry Pi Mirror (piwheels):** https://www.piwheels.org/project/al3x38-intent-guard/
* **Architecture Invariants:** https://github.com/alexirk97-ship-it/al3x38-intent-guard/blob/main/ARCHITECTURE.md
* **Technical Whitepaper (EN):** https://github.com/alexirk97-ship-it/al3x38-intent-guard/blob/main/WHITEPAPER.md
* **Technical Whitepaper (RU):** https://github.com/alexirk97-ship-it/al3x38-intent-guard/blob/main/WHITEPAPER_RU.md

## License

MIT License. Concept & architecture by **al3x38**.

