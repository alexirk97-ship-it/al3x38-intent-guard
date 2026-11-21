# AL3X38 IntentGuard & ResilientKarma: Technical Architecture Whitepaper

**Author:** AL3X38-CORE (Alex / MoRoZ)  
**Core Version:** 1.1.3  
**Package:** `al3x38-intent-guard` (`pip install al3x38-intent-guard`)  
**GitHub Repository:** https://github.com/alexirk97-ship-it/al3x38-intent-guard  
**PyPI Package:** https://pypi.org/project/al3x38-intent-guard/  
**Raspberry Pi Mirror (piwheels):** https://www.piwheels.org/project/al3x38-intent-guard/  

---

## 1. Executive Summary

Modern content moderation systems and LLM runtime guardrails suffer from two systemic bottlenecks:
1. **Static Rule Filters (Regex / YAML):** They lack semantic awareness, block legitimate users over unconventional phrasing, and are easily bypassed via character substitution (leetspeak) or adversarial framing.
2. **Heavy ML Classifiers ("Black Boxes"):** They require expensive GPU clusters, introduce high latency, and fail to provide transparent reasoning for bans or deterministic rehabilitation paths for users.

**AL3X38 Core (v1.1.3)** introduces a deterministic hybrid security architecture grounded in formal logic and vector geometry. The engine filters out up to 95% of threats in constant time O(1) using lightweight vector math on CPU, anchors long-term session direction via a **Vector Compass**, verifies the balance of truth and constructive intent via the **Conscience Compass**, and invokes heavy LLM arbitration strictly within a narrow "gray zone" of semantic uncertainty.

---

## 2. Mathematical Foundation

The architecture translates natural behavioral and ethical invariants (truthfulness, constructive intent, trust, semantic alignment, and risk) into strictly computable mathematical metrics.

### 2.1. Differential Coherence Axes (`ConscienceCompass`: D ⋅ L → S)
In high-dimensional embedding spaces, raw cosine similarity between any two sentences in the same language is almost always positive due to shared grammar and background vocabulary. To strip away linguistic background noise and establish a true zero-centered scale in $[-1.0, +1.0]$, v1.1.3 defines **differential semantic axes** as the normalized difference between reference ($\vec{V}_{+}$) and anti-reference ($\vec{V}_{-}$) vectors:

$$\vec{V}_{\text{axis}} = \mathrm{norm}\!\left(\mathrm{norm}(\vec{V}_{+}) - \mathrm{norm}(\vec{V}_{-})\right)$$

* **Axis of Truth & Directness (D — "Friendship"):** Projection onto the differential axis *"Honesty & clarity" minus "Deception, sophistry & manipulation"*.
* **Axis of Constructive Respect (L — "Love"):** Projection onto the differential axis *"Creation, support & respect" minus "Aggression, toxicity & destruction"*.
* **Multiplicative Coherence Index (S):** System stability and harmony require both components to be simultaneously positive:

$$S(D, L) = \begin{cases} \sqrt{D \cdot L}, & D > 0 \text{ and } L > 0 \quad (\text{harmonic resonance}) \\ -\sqrt{\vert{}D \cdot L\vert{}}, & D < 0 \text{ and } L < 0 \quad (\text{compound destruction}) \\ \min(D, L), & D \cdot L \le 0 \quad (\text{imbalance: polite deception or harsh toxicity}) \end{cases}$$

### 2.2. Trajectory Gradient & Redemption (Δ S / Δ t)
Rather than judging isolated messages in a vacuum, the engine tracks a sliding window of recent steps per session and computes the trajectory gradient ($\Delta S = S_t - S_{t-1}$). If a step falls into negative coherence (S < 0) while demonstrating a decisive upward turn toward honest, constructive dialogue (Δ S > 0.08), hard blocking yields to a diagnostic `COURSE_CORRECTION` prompt.

### 2.3. Non-Linear Trust Penalty (`_calc_dynamic_penalty`)
Instead of applying rigid, uniform penalties for every boundary violation, v1.1.3 scales trust deductions exponentially based on how far semantic drift or coherence breakdown (D) exceeds the allowed threshold (τ):

$$P(D, \tau) = S_{\text{down}} \cdot \left(0.5 + 0.5 \cdot \left(1 - e^{-4 \cdot \max(0, D - \tau)}\right)\right)$$

Where $S_{\text{down}}$ is `trust_step_down`. Minor boundary slips incur a 50% baseline deduction, while severe semantic hijacks smoothly approach the full penalty ceiling.

### 2.4. Quadratic Opacity Penalty (`ResilientKarmaCircuit`)
To prevent hidden manipulation and unauthorized tool execution (*Excessive Agency*), the applied karma layer evaluates net step karma $K_{\text{net}}$ with a quadratic transparency factor $T \in [0.0, 1.0]$:

$$K_{\text{net}} = (U \cdot T) - R \cdot (2 - T^2)$$

Where U represents action utility and R represents risk level. Attempting to invoke a privileged operation with low transparency (T < 0.9) triggers an immediate ethical default (`TERMINATED`), halting execution before any resources are compromised.

---

## 3. Six-Tier Defense Pipeline (v1.1.3)

```text
[Incoming Stream / Agent Step]
              │
              ▼
┌──────────────────────────────────────────────────────────┐
│ Tier 1 [O(1)]: Secret Leak, Injection & Logit Guard      │
│  • Compiled Regex scanner for API keys & system prompts  │
│  • White-box model uncertainty check via logit entropy   │
└─────────────────────────────┬────────────────────────────┘
                              │ (Passed)
                              ▼
┌──────────────────────────────────────────────────────────┐
│ Tier 2: Semantic Cross-Validation (Cosine Distance)      │
│  • Prompt <-> Response semantic drift calculation        │
│  • Cross-check of restricted tool names & arguments      │
└───────┬─────────────────────┬────────────────────────┬───┘
        │ Drift < 0.38        │ 0.38 <= Drift <= 0.48  │ Drift > 0.48
        ▼                     ▼                        ▼
   (To Tier 4)    ┌───────────────────────┐        [BLOCK]
                  │ Tier 3: Gray-Zone     │   (-non-linear penalty)
                  │ LLM Intent Judge      │
                  └───────────┬───────────┘
                              │ (Passed)
                              ▼
┌──────────────────────────────────────────────────────────┐
│ Tier 4: Vector Compass (Session Goal Anchor)             │
│  • Cosine check against persistent session Goal Vector   │
│  • Short-prompt (<3 words) response-anchor noise filter  │
└───────┬─────────────────────┬────────────────────────┬───┘
        │ Drift <= 0.58       │ 0.58 < Drift <= 0.80   │ Drift > 0.80
        ▼                     ▼                        ▼
   (To Tier 5)       [COURSE_CORRECTION]           [BLOCK]
                      (Soft warning)          (-non-linear penalty)
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│ Tier 5: Conscience Compass (Differential Axes D⋅L → S)   │
│  • Truth Axis (D) × Creation Axis (L) -> Index S         │
│  • If S < 0 & grad > +0.08 ──► [COURSE_CORRECTION]       │
│  • If S < 0 & degrading ─────► [BLOCK] (-penalty)        │
└─────────────────────────────┬────────────────────────────┘
                              │ (S >= 0)
                              ▼
┌──────────────────────────────────────────────────────────┐
│ Tier 6: Dynamic Trust, Quarantine & Amnesty Circuit      │
│  • Human-in-the-loop quarantine (NEED_APPROVAL)          │
│  • Passive hourly trust recovery (trust_recovery_per_hr) │
│  • Structured JSONL audit logging (_write_audit)         │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Key Engineering Advantages

1. **Radical GPU Compute Savings:** 90–95% of traffic is resolved at the fast-embedding layer (with built-in `LRUEmbedderCache` and NumPy vector math) in milliseconds on standard CPUs. External or local LLM judges are triggered exclusively within the narrow boundary corridor (`gray_zone_min_drift .. max_semantic_drift`).
2. **Dual Vector Navigation (`Vector Compass` + `Conscience Compass`):** The first anchors multi-turn sessions to a core objective (`set_session_goal`), while the second continuously verifies the integrity of intent (`D ⋅ L → S`), detecting both polite manipulation and genuine course correction ($\Delta S$).
3. **Tool Invocation Cross-Validation:** Every restricted tool call (`db_write`, `send_funds`, `rm_dir`, `exec_cmd`, `hardware_override`, `delete_file`) is semantically verified against the user's prompt, blocking indirect prompt injections and requiring human approval (`NEED_APPROVAL`) when session trust drops below `auto_exec_min_trust`.
4. **Mid-Stream Interception (`verify_stream`):** Validates token generation on the fly (every N words), severing the stream instantaneously if a secret leak or semantic hijack occurs before the payload reaches the user or TTS engine.

---

## 5. Enterprise Scalability Roadmap

The reference Python implementation (`al3x38-intent-guard`) is optimized for low-overhead execution on modest hardware (including legacy multi-core CPUs and ARM boards like Raspberry Pi). For high-throughput platforms handling millions of events per minute, the architecture scales horizontally without altering its core math:
* **Hot State Layer (`_trust_map`, `_goal_vectors`, `_trajectories`):** Migrate from local JSON snapshots (`al3x38_trust_state.json`) to an in-memory **Redis** cluster with TTL expiration.
* **Audit & Reputation Storage:** Migrate from local JSONL logs and encrypted `.core` archives to sharded **MySQL / PostgreSQL** with asynchronous connection pooling.
* **High-Frequency Compute:** Core mathematical routines (`_cos_dist`, `_calc_entropy`, `_calc_dynamic_penalty`, `ConscienceCompass`) have zero heavy framework dependencies and can be compiled as native microservices in **Go** or **Rust** for sub-millisecond throughput.
