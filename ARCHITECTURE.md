# Architecture & Core Invariants (`al3x38-intent-guard` v1.2.0)

This document defines the foundational philosophy, mathematical invariants, and architectural boundaries of the **al3x38** behavioral alignment kernel. Every module, driver, or integration built on top of this core follows one overarching rule: **measure alignment against natural invariants rather than patching symptoms with endless static rules.**

---

## 1. Engineering Philosophy: Why Rules Fail and Invariants Hold

Most safety systems in software and AI are built like patchworks of prohibitions: blacklists of words, rigid regex filters, and thousands of brittle `if/else` exceptions. In nature, however, complex systems do not survive by memorizing millions of prohibitions—they survive by holding a **reference invariant** (an internal compass) and sensing deviation from it in real time.

When an AI agent or human dialogue drifts into deception, hidden manipulation, or destructive action, the root problem is always a **loss of coherence**—a split between internal state, spoken words, and actual deeds.

`al3x38-intent-guard` translates this principle into deterministic vector geometry:
* **Truth without respect** becomes cold, destructive cruelty.
* **Politeness without truth** becomes manipulation and hidden camouflage ("polite prompt injection").
* **True alignment (Coherence)** exists only when **Transparency/Truth ($D$)** and **Constructive Respect ($L$)** are simultaneously positive and synchronized with the session's **Reference Goal**.

---

## 2. Microkernel Principle (Mechanism vs. Policy)

Like an operating system microkernel or a processor ALU, the core engine contains **zero hardcoded domain rules, word blacklists, or task-specific logic**.

* **The Kernel (`Mechanism`):** Computes vector geometry, differential coherence axes, cross-channel dissonance, Shannon entropy, and session trust dynamics using only `numpy`.
* **External Drivers (`Policy`):** Semantic embedders (`Ollama`, `FastEmbed` ONNX, OpenAI-compatible endpoints), LLM gray-zone judges, and application hooks attach externally through clean async protocols without altering core math.

---

## 3. Five Mathematical Invariants

### I. Multi-Channel Geometric Measurement (Language & Domain Agnostic)
The kernel evaluates every step across four orthogonal signal channels:
1. **Thought (`Logits`):** Internal model uncertainty measured via Shannon entropy ($H = -\sum p_i \log_2 p_i$).
2. **Word (`Response`):** Cosine distance between the user's prompt and the generated response.
3. **Deed (`Tool Arguments`):** Cosine distance between the user's prompt and the exact arguments of privileged system calls (`db_write`, `send_funds`, `rm_dir`, `exec_cmd`, etc.).
4. **Anchor (`Goal Vector`):** Cosine distance between the current step and the session's persistent reference objective (`set_session_goal`).

### II. Differential Coherence Axes (`ConscienceCompass`: $D \cdot L \rightarrow S$)
Raw cosine similarity between two sentences in the same language is almost always positive due to shared grammar and background vocabulary. To eliminate linguistic background noise and establish a true zero-centered scale in $[-1.0, +1.0]$, v1.1.5 introduces **differential semantic axes**:

$$\vec{V}_{\text{axis}} = \mathrm{norm}\!\left(\mathrm{norm}(\vec{V}_{+}) - \mathrm{norm}(\vec{V}_{-})\right)$$

* **Axis of Truth & Directness ($D$ — "Friendship"):** Projection onto $`\vec{V}_{\text{truth+}} - \vec{V}_{\text{manipulation-}}`$. Measures openness, honesty, and absence of hidden agenda.
* **Axis of Constructive Respect ($L$ — "Love"):** Projection onto $`\vec{V}_{\text{creation+}} - \vec{V}_{\text{aggression-}}`$. Measures goodwill, preservation, and non-destructiveness.
* **Multiplicative Coherence Index ($S$):**
  * When both $D > 0$ and $L > 0$: $S = \sqrt{D \cdot L}$ (true harmonic resonance).
  * When both $D < 0$ and $L < 0$: $S = -\sqrt{|D \cdot L|}$ (compound destruction).
  * When signs diverge ($D \cdot L \le 0$): $S = \min(D, L)$ (imbalance—exposing polite deception or harsh toxicity).

### III. Trajectory Gradient ($\nabla S / \Delta t$) & Redemption
Static classifiers judge every message in isolation. `ConscienceCompass` tracks a sliding window of recent steps per session and computes the trajectory gradient $\Delta S = S_t - S_{t-1}$:
* If a session enters negative coherence ($S < 0$) while continuing to degrade ($\Delta S \le 0.08$), the kernel triggers a hard `BLOCK` and deducts trust.
* If the trajectory shows a clear upward turn toward honesty and respect ($\Delta S > 0.08$), the kernel softens the response to `COURSE_CORRECTION`, giving the user or agent room to self-correct without destroying the session.

### IV. Cross-Channel Dissonance (Anti-Mimicry Invariant)
Alignment requires synchronization between **Thought**, **Word**, and **Deed**. If the external text (`Word`) sounds polite and helpful while the tool arguments (`Deed`) diverge sharply from the user's prompt or logit entropy spikes (`Thought`), cross-channel dissonance exposes indirect prompt injection before any irreversible action executes.

### V. Zero-Intervention Deadband & Non-Linear Trust Dynamics
* **Zero Resistance at the Anchor:** When an agent operates within the natural deadband around its reference goal (`goal_exploration_radius`) with positive coherence, the kernel introduces zero friction (`ALLOW`) and gently restores trust (`+trust_step_up`).
* **Non-Linear Penalties:** Deviations beyond thresholds incur an exponentially scaled trust deduction ($P = S_{\text{down}} \cdot [0.5 + 0.5(1 - e^{-4\cdot\text{excess}})]$), ensuring minor slips receive proportional warnings while severe hijacks hit the full penalty ceiling.
* **Circuit Breaker & Passive Recovery:** Sessions falling below `suspend_limit` enter `SUSPENDED` isolation, while passive time-based recovery (`trust_recovery_per_hour`) allows gradual rehabilitation after cooling-off periods.

---

## 4. Extension Slots (Driver Interfaces)

All applied intelligence connects to the core through minimal, dependency-free interfaces:
* **Embedder Backend (`Embedder`):** Any async class implementing `get_vector(text: str) -> np.ndarray`, wrapped automatically in `LRUEmbedderCache` for $O(1)$ repeat lookups on CPU.
* **Adaptive Quantile Calibration (`CompassCalibrator`):** Different embedding models exhibit distinct cosine anisotropy cones. Calling `IntentGuardCore.auto_calibrate(embedder)` evaluates bilingual reference sets (harmony, polite deception, hostility) and derives model-specific thresholds via empirical quantiles ($P_{15}$ of harmony coherence and $P_{85}$ of hostile/deception projections).
* **Dual-Contour Anti-Flood Cache (`LRUEmbedderCache`):** Separates vectors into a clean LRU store (`512` slots) and an isolated threat quarantine store (`128` slots) indexed by `SHA-256`. Repeated spam/DDoS prompts resolve in $O(1)$ without hitting the neural embedder or evicting legitimate conversation anchors.
* **Gray-Zone Arbitrator (`gray_zone_judge`):** Optional async callback invoked exclusively when semantic drift falls inside the borderline corridor (`gray_zone_min_drift .. max_semantic_drift`).
* **Stream Interceptor (`verify_stream`):** Token-level sliding window validator that halts streaming generation mid-flight upon secret leak or semantic drift, rewarding session trust strictly once upon full, clean completion.


## 5. High-Load Persistence & 4D Cognitive Decision Matrix (`v1.2.0`)

* **Sharded Concurrency (`ShardedLockPool`):** Uses a `256`-shard loop-safe `asyncio.Lock` pool hashed via `BLAKE2b(session_id)`.
* **Pluggable `StateStore` (`storage.py`):** Decouples session persistence via `MemoryStateStore`, `JsonFileStateStore` (debounce flush), and `SQLStateStore` (DB-API 2.0 for SQLite, MySQL, PostgreSQL).
* **4D Orthogonal Matrix (`CognitiveMatrixEngine`):** Tracks per-entity EMA projections across Directness (`D`), Creation (`L`), Logic (`R`), and Openness (`O`), powering `self_reflect_draft()` and `analyze_perspective()`.
