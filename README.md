**English** | [Русский](README_RU.md)

# al3x38-intent-guard

[![PyPI version](https://img.shields.io/pypi/v/al3x38-intent-guard.svg)](https://pypi.org/project/al3x38-intent-guard/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**`al3x38-intent-guard`** is an asynchronous behavioral guardrail and intent-to-action verification layer for LLM agents in Python. It checks whether an agent's text response and system tool calls (`Function Calling`) actually match the user's original intent.

---

## Why I built this

Most AI guardrails rely on static regex blacklists or heavy external classifier models. They catch obvious bad words, but fail against **Indirect Prompt Injection** and subtle behavioral drift—where an LLM writes a polite, relevant reply to the user while silently triggering a privileged tool (`rm_dir`, `db_write`, `send_funds`) with unrelated arguments in the background.

Instead of static rules, I modeled `IntentGuardCore` after how **human trust and cognitive self-control** work in real life:
* **Initial Trust & Reputation (`trust_map`):** Every session starts with a baseline trust score (`0.8`). Normal behavior builds it up slowly (`+0.02`), while suspicious drift or manipulation drops it sharply (`-0.25`).
* **Semantic Drift Detection:** Measures cosine distance between the user prompt, the model's response, and the exact arguments of any restricted tool call.
* **Internal Uncertainty (`White-Box Entropy`):** When `logprobs` are available, calculates Shannon entropy to detect internal confusion or adversarial steering before an action executes.
* **The "Gray Zone" & Human-in-the-Loop:** If a step falls into a borderline drift range (`gray_zone_min_drift .. max_semantic_drift`), it can consult an async `gray_zone_judge` or pause execution for human confirmation (`NEED_APPROVAL`).
* **Time-Based Amnesty (`Trust Recovery`):** Just like human emotions cooling down over time, session trust gradually recovers (`trust_recovery_per_hour`) if no further violations occur—unless the session crosses the hard `SUSPENDED` threshold.

---

## Architecture Pipeline

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
        3. Logit Entropy (White-Box) & Gray-Zone Judge
           └─► High drift or high entropy ──► [BLOCK] (-trust)
                        │
                        ▼
        4. Low-Trust Quarantine Check
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

With built-in offline CPU embeddings (no Ollama or external API required):
```bash
pip install al3x38-intent-guard[offline]
```

---

## Quick Start

### 1. Local Ollama Integration
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

    # Raw payload from Ollama (/api/chat)
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

### 2. Fully Offline Mode (CPU)
```python
from al3x38_intent_guard import IntentGuardCore, OfflineFastEmbedder

embedder = OfflineFastEmbedder(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
guard = IntentGuardCore(embedder)
```

### 3. OpenAI / LM Studio / vLLM Compatible APIs
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

## Key Features (v1.1.0)

* **Real-Time Stream Guard (`verify_stream`):** Inspects token streams on the fly (every N words) and aborts generation mid-sentence if a secret leak or semantic drift occurs.
* **Secret & Prompt Leak Filter (`Secret Guard`):** Built-in regex scanner (`enable_secret_guard=True`) catches accidental exposure of API keys (`sk-...`, `hf_...`, `pypi-...`, `AKIA...`) and system prompt fragments.
* **Hybrid Gray-Zone Arbitration:** Offloads ambiguous edge cases to a custom async `gray_zone_judge` callback without slowing down routine requests.
* **Persistent State & JSONL Audit Logs:** Saves session trust state safely across restarts and logs all verdicts (`ALLOW`, `NEED_APPROVAL`, `BLOCK`, `SUSPENDED`) with drift metrics and live stats via `guard.get_stats()`.

---

## Links

* **GitHub:** [alexirk97-ship-it/al3x38-intent-guard](https://github.com/alexirk97-ship-it/al3x38-intent-guard)
* **PyPI:** [al3x38-intent-guard](https://pypi.org/project/al3x38-intent-guard/)

## License

MIT License. Concept & architecture by **al3x38**.
