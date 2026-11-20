**English** | [Русский](README_RU.md)

# al3x38-intent-guard

[![PyPI version](https://img.shields.io/pypi/v/al3x38-intent-guard.svg)](https://pypi.org/project/al3x38-intent-guard/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/personalized-badge/al3x38-intent-guard?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/al3x38-intent-guard)

**`al3x38-intent-guard`** is an asynchronous behavioral guardrail and vector coherence compass for LLM agents in Python. Instead of relying on brittle keyword blacklists, it verifies **deep semantic alignment** between the user's original intent, the session's long-term goal, the model's response, and privileged system actions (`Function Calling`).

---

## Philosophy & Design Principle

Most AI safety tools are built like endless walls of prohibitions: static regex filters, word blacklists, and heavy external classifier models. In living systems, however, stability does not come from memorizing millions of prohibitions—it comes from holding a **reference invariant** (an internal compass) and sensing deviation from harmony in real time.

At the core of `al3x38-intent-guard` lies a strict, geometric model of behavioral coherence:
* **Truth without respect (D > 0, L < 0)** turns into cold, destructive hostility.
* **Politeness without truth (D < 0, L > 0)** becomes hidden manipulation and camouflage ("polite prompt injection," where an agent smiles at the user while silently attempting to delete files).
* **True alignment (Coherence, \(S = \sqrt{D \cdot L}\))** exists only when **Transparency/Truth (D)** and **Constructive Respect (L)** are simultaneously positive and aligned with the session's **Reference Goal**.

The kernel translates this principle into fast, dependency-free vector math on `numpy`, running in milliseconds on standard CPUs without expensive GPU clusters.

---

## Defense Pipeline (v1.1.3)

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

---

## What's New in v1.1.3

* **Conscience Compass (`ConscienceCompass` & `ConscienceState`):** Introduced differential semantic axes measuring **Truth/Directness ($D$)** and **Constructive Respect ($L$)**. By subtracting anti-anchor vectors from reference anchors ($\vec{V}_{+} - \vec{V}_{-}$), the kernel strips away shared linguistic background noise and exposes "polite deception" or hidden manipulation ($S = \sqrt{D \cdot L}$).
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

