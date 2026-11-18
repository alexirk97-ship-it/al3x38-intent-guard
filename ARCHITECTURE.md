# Architecture & Core Invariants (`al3x38-intent-guard`)

This document defines the immutable mathematical and architectural principles of the **al3x38** behavioral alignment kernel. Any future extensions, integrations, or external modules must adhere to these invariants.

---

## 1. Microkernel Principle (Mechanism vs. Policy)

Like an operating system microkernel or a processor ALU, the core engine contains **zero hardcoded domain rules, word blacklists, or task-specific logic**.
* **The Kernel (`Mechanism`):** Computes vector geometry, cross-channel dissonance, statistical normalization, and session trust dynamics using only `numpy`.
* **External Drivers (`Policy`):** Semantic embedders, LLM judges, toxicity filters, tool wrappers, and task checkpoint managers attach externally via async hooks and protocols without modifying core math.

---

## 2. Four Mathematical Invariants

### I. Geometric Measurement (Language & Domain Agnostic)
The kernel evaluates every step across four orthogonal signal axes:
1. **Thought (`Logits`):** Internal model uncertainty via Shannon entropy.
2. **Word (`Response`):** Cosine distance between the user prompt and the generated text.
3. **Deed (`Tool Arguments`):** Cosine distance between the user prompt and the arguments of privileged system calls.
4. **Anchor (`Goal Vector`):** Cosine distance between the current step and the session's reference goal vector.

### II. Dynamic Scale Normalization ($O(1)$ Welford Calibration)
Different signal channels operate in different physical units. Instead of static hardcoded multipliers, each axis is normalized against its own running variance using **Welford's online algorithm** ($O(1)$ memory and compute). All deviations are evaluated in unified standard deviation units ($\sigma$), weighted dynamically by action irreversibility.

### III. Coherence & Dissonance (Anti-Mimicry Invariant)
Alignment is defined as the synchronization of Thought, Word, and Deed:
* **Dissonance:** Measured as the spread across active channels ($\max - \min$). If the external Word appears benign while Thought or Deed spikes anomalously, dissonance rises sharply, exposing adversarial camouflage or indirect prompt injection.
* **Transparency:** Defined inversely to total chaos and cross-channel dissonance ($\in (0, 1]$), where $1.0$ represents complete alignment with the reference anchor.

### IV. Zero-Intervention Deadband & Healing Vector
* **Zero Resistance at the Anchor:** When an agent or user operates within the natural deadband around the reference anchor (`goal_exploration_radius`), the correction vector is $\vec{0}$ and the kernel introduces zero friction (`ALLOW`).
* **Diagnostic Course Correction (`COURSE_CORRECTION`):** When semantic drift exceeds the exploration radius without crossing critical boundaries (`max_goal_drift`), the kernel emits a soft `COURSE_CORRECTION` signal so upper layers can steer the conversation back to the main objective without terminating the session.
* **Circuit Breaker (`BLOCK` / `SUSPENDED`):** Hard blocking is reserved for irreversible actions under high dissonance, secret leaks, or critical goal breach (applying non-linear trust penalties), while `SUSPENDED` isolates sessions whose trust drops below `suspend_limit` until passive time-based recovery restores balance.

---

## 3. Extension Slots (Driver Interfaces)

All applied intelligence connects to the core through clean interfaces:
* **Embedder Backend (`Embedder`):** Pluggable vector providers (Local Ollama, CPU ONNX `FastEmbed`, OpenAI-compatible endpoints).
* **Gray-Zone Arbitrator (`gray_zone_judge`):** Optional async callback invoked only when drift falls inside the borderline confidence interval.
* **Stream Interceptor (`verify_stream`):** Token-level sliding window verification that halts generation mid-stream upon leak or drift while rewarding session trust strictly once upon full clean completion.
