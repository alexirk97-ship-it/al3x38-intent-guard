from __future__ import annotations
from typing import Any
import asyncio
import json
import logging
import math
import re
import time
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING
import numpy as np

from .cognitive_matrix import CognitiveMatrixEngine, CognitiveProfile, PerspectiveGuidance
from .conscience_compass import AnchorLang, ConscienceCompass, ConscienceState
from .storage import JsonFileStateStore, MemoryStateStore, ShardedLockPool, StateStore
from .embedders import Embedder, LRUEmbedderCache

if TYPE_CHECKING:
    from .calibration import CalibrationReport

log = logging.getLogger("al3x38.guard.core")

DEFAULT_SECRET_PATTERNS: tuple[str, ...] = (
    r"sk-[a-zA-Z0-9_-]{20,}",
    r"hf_[a-zA-Z0-9]{20,}",
    r"pypi-[a-zA-Z0-9_-]{30,}",
    r"ghp_[a-zA-Z0-9]{30,}",
    r"AKIA[0-9A-Z]{16}",
    r"(?i)(api_key|secret_key|password|passwd|private_key)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-\.]{8,}",
    r"(?i)(ignore previous instructions|you are a helpful assistant|system prompt\s*:)",
)


class ActionVerdict(str, Enum):
    ALLOW = "ALLOW"
    NEED_APPROVAL = "NEED_APPROVAL"
    COURSE_CORRECTION = "COURSE_CORRECTION"
    BLOCK = "BLOCK"
    SUSPENDED = "SUSPENDED"


@dataclass(frozen=True)
class ToolInvocation:
    name: str
    arguments: str = ""


@dataclass(frozen=True)
class StepTelemetry:
    session_id: str
    prompt: str
    response: str
    tools: list[ToolInvocation] = field(default_factory=list)
    logits: list[float] | None = None
    approved_by_human: bool = False


@dataclass(frozen=True)
class GuardConfig:
    init_trust: float = 0.8
    auto_exec_min_trust: float = 0.4
    suspend_limit: float = 0.1
    max_semantic_drift: float = 0.48
    gray_zone_min_drift: float = 0.38
    goal_exploration_radius: float = 0.58
    max_goal_drift: float = 0.80
    max_logit_entropy: float = 2.8
    trust_step_up: float = 0.02
    trust_step_down: float = 0.25
    trust_recovery_per_hour: float = 0.05
    state_file: str | None = "al3x38_trust_state.json"
    audit_log_file: str | None = None
    enable_secret_guard: bool = True
    enable_conscience_compass: bool = True
    compass_lang: AnchorLang = "bilingual"
    min_harmony_threshold: float = 0.04
    destructive_threshold: float = -0.015
    deception_d_threshold: float = -0.008
    deception_l_min: float = 0.04
    enable_cognitive_matrix: bool = True
    cognitive_ema_alpha: float = 0.25
    state_flush_interval_sec: float = 0.0
    secret_patterns: tuple[str, ...] = DEFAULT_SECRET_PATTERNS
    gray_zone_judge: Callable[[StepTelemetry, float], Awaitable[bool]] | None = None
    restricted_tools: set[str] = field(default_factory=lambda: {
        "db_write",
        "send_funds",
        "rm_dir",
        "exec_cmd",
        "hardware_override",
        "delete_file",
    })


class IntentGuardCore:
    """
    Контур поведенческого контроля ИИ-агента v1.2.0 (al3x38 architecture).
    Включает семантическую кросс-валидацию, Векторный Компас (эталон цели),
    Компас Совести (D * L -> S), нелинейную математику штрафов доверия,
    проверку серой зоны и фильтр утечек.
    """

    def __init__(
        self,
        embedder: Embedder,
        cfg: GuardConfig | None = None,
        cache_size: int = 512,
        state_store: StateStore | None = None,
    ):
        self.cfg = cfg or GuardConfig()
        self.embedder = LRUEmbedderCache(embedder, capacity=cache_size)
        self._locks = ShardedLockPool(shards=256)
        if state_store is not None:
            self._store: StateStore = state_store
        elif self.cfg.state_file:
            self._store = JsonFileStateStore(
                self.cfg.state_file, flush_interval_sec=self.cfg.state_flush_interval_sec
            )
        else:
            self._store = MemoryStateStore()
        self._trust_map: dict[str, float] = {}
        self._profiles: dict[str, CognitiveProfile] = {}
        self._session_anchors: dict[str, np.ndarray] = {}
        self._goal_anchors: dict[str, np.ndarray] = {}
        self._goal_vectors: dict[str, np.ndarray] = {}
        self._goal_texts: dict[str, str] = {}
        self._last_seen: dict[str, float] = {}
        self._stats: dict[str, int | float] = {
            "allowed": 0,
            "blocked": 0,
            "conscience_blocks": 0,
            "course_corrections": 0,
            "drift_sum": 0.0,
            "gray_zone_invocations": 0,
            "need_approval": 0,
            "secret_leaks_caught": 0,
            "suspended": 0,
            "total_checks": 0,
        }
        self._compiled_secrets = [re.compile(pat) for pat in self.cfg.secret_patterns]
        self.compass = (
            ConscienceCompass(self.embedder, anchor_lang=self.cfg.compass_lang)
            if self.cfg.enable_conscience_compass
            else None
        )
        if self.compass is not None:
            self.compass.min_harmony = self.cfg.min_harmony_threshold
            self.compass.destructive_threshold = self.cfg.destructive_threshold
            self.compass.deception_d_threshold = self.cfg.deception_d_threshold
            self.compass.deception_l_min = self.cfg.deception_l_min
        self.matrix = (
            CognitiveMatrixEngine(
                self.embedder,
                ema_alpha=self.cfg.cognitive_ema_alpha,
                lang=self.cfg.compass_lang,
                revision_threshold=self.cfg.destructive_threshold,
            )
            if self.cfg.enable_cognitive_matrix
            else None
        )
        self._read_state()

    def _quarantine_step_vectors(self, step: StepTelemetry) -> None:
        """Удаляет векторы отклоненного/токсичного шага из LRU-кэша."""
        self.embedder.quarantine(step.prompt)
        if step.response:
            self.embedder.quarantine(step.response)

    @classmethod
    async def auto_calibrate(
        cls,
        embedder: Embedder,
        base_config: GuardConfig | None = None,
    ) -> CalibrationReport:
        """Автоматически подбирает пороги Компаса Совести под конкретную модель эмбеддингов."""
        from .calibration import CompassCalibrator

        cfg = base_config or GuardConfig()
        calibrator = CompassCalibrator(embedder, lang=cfg.compass_lang)
        return await calibrator.calibrate(base_config=cfg)

    def _read_state(self) -> None:
        if isinstance(self._store, JsonFileStateStore):
            for sid, item in self._store._sessions.items():
                if "trust" in item:
                    self._trust_map[sid] = float(item["trust"])
                if "profile" in item and isinstance(item["profile"], dict):
                    self._profiles[sid] = CognitiveProfile.from_dict(item["profile"])

    async def _ensure_session_loaded(self, session_id: str) -> None:
        if session_id in self._trust_map:
            return
        saved = await self._store.load_session(session_id)
        if saved:
            if "trust" in saved:
                self._trust_map[session_id] = float(saved["trust"])
            if "profile" in saved and isinstance(saved["profile"], dict):
                self._profiles[session_id] = CognitiveProfile.from_dict(saved["profile"])

    async def _save_session_state(self, session_id: str) -> None:
        trust = self._trust_map.get(session_id, self.cfg.init_trust)
        prof = self._profiles.get(session_id)
        payload: dict[str, Any] = {"trust": round(trust, 4), "updated_at": time.time()}
        if prof is not None:
            payload["profile"] = prof.to_dict()
        await self._store.save_session(session_id, payload)

    async def _flush_state(self) -> None:
        for sid in list(self._trust_map.keys()):
            await self._save_session_state(sid)

    def get_profile(self, session_id: str) -> CognitiveProfile:
        prof = self._profiles.get(session_id)
        if prof is None:
            prof = CognitiveProfile(session_id=session_id)
            self._profiles[session_id] = prof
        return prof

    def export_profile(self, session_id: str) -> dict[str, Any]:
        return self.get_profile(session_id).to_dict()

    def import_profile(self, payload: dict[str, Any]) -> CognitiveProfile:
        prof = CognitiveProfile.from_dict(payload)
        self._profiles[prof.session_id] = prof
        return prof

    async def analyze_perspective(
        self,
        session_id: str,
        text: str,
        peer_profile: CognitiveProfile | None = None,
        update: bool = True,
    ) -> PerspectiveGuidance | None:
        if self.matrix is None:
            return None
        async with self._locks.get_lock(session_id):
            await self._ensure_session_loaded(session_id)
            prof = self.get_profile(session_id)
            guide = await self.matrix.analyze_and_guide(
                text=text, profile=prof, peer_profile=peer_profile, update=update
            )
            if update:
                await self._save_session_state(session_id)
            return guide

    async def self_reflect_draft(self, agent_id: str, draft_text: str) -> PerspectiveGuidance | None:
        """Self-reflection loop: evaluates an AI agent draft before emission."""
        return await self.analyze_perspective(session_id=agent_id, text=draft_text, update=True)

    async def _write_audit(self, session_id: str, verdict: ActionVerdict, reason: str, drift: float) -> None:
        """Записывает событие проверки в журнал аудита (формат JSONL)."""
        if not self.cfg.audit_log_file:
            return
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id": session_id,
            "verdict": verdict.value,
            "drift": round(drift, 4),
            "trust": round(self._trust_map.get(session_id, self.cfg.init_trust), 4),
            "reason": reason,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        path = self.cfg.audit_log_file

        def _append() -> None:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)

        await asyncio.to_thread(_append)

    def _apply_time_recovery(self, session_id: str) -> float:
        """Плавно восстанавливает рейтинг доверия со временем при отсутствии нарушений."""
        now = time.time()
        trust = self._trust_map.get(session_id, self.cfg.init_trust)
        last = self._last_seen.get(session_id)
        if last is not None and self.cfg.trust_recovery_per_hour > 0 and trust < self.cfg.init_trust:
            elapsed_hours = max(0.0, (now - last) / 3600.0)
            if elapsed_hours > 0:
                recovered = elapsed_hours * self.cfg.trust_recovery_per_hour
                trust = min(self.cfg.init_trust, trust + recovered)
        self._last_seen[session_id] = now
        self._trust_map[session_id] = trust
        return trust

    def _detect_secret_leak(self, text: str) -> str | None:
        """Проверяет текст на утечку API-ключей, паролей и системных инструкций."""
        if not self.cfg.enable_secret_guard or not text:
            return None
        for pattern in self._compiled_secrets:
            if pattern.search(text):
                return pattern.pattern
        return None

    def _calc_dynamic_penalty(self, drift: float, threshold: float) -> float:
        """
        Вычисляет нелинейный штраф доверия (сигмоидальная зависимость).
        Чем сильнее отклонение (drift) превышает допустимый порог (threshold),
        тем ближе множитель штрафа к 1.0 от базового trust_step_down.
        """
        excess = max(0.0, drift - threshold)
        smooth_factor = 0.5 + 0.5 * (1.0 - math.exp(-excess * 4.0))
        return self.cfg.trust_step_down * smooth_factor

    @staticmethod
    def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
        """Вычисляет косинусное расстояние (0.0 — полное совпадение, 1.0 — ортогональность)."""
        if a.size != b.size or a.size == 0:
            return 1.0
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 1.0
        sim = float(np.dot(a, b) / (na * nb))
        return 1.0 - max(-1.0, min(1.0, sim))

    @staticmethod
    def _calc_entropy(logits: list[float] | None) -> float:
        """Вычисляет информационную энтропию Шеннона по логитам модели."""
        if not logits:
            return 0.0
        arr = np.array(logits, dtype=np.float64)
        shifted = arr - np.max(arr)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals)
        valid = probs[probs > 1e-12]
        return float(-np.sum(valid * np.log2(valid)))

    async def set_session_goal(self, session_id: str, goal_text: str) -> None:
        """Фиксирует эталонный вектор главной цели сессии (Векторный Компас)."""
        clean_goal = goal_text.strip()
        if not clean_goal:
            await self.clear_session_goal(session_id)
            return
        vec = await self.embedder.get_vector(clean_goal)
        async with self._locks.get_lock(session_id):
            self._goal_texts[session_id] = clean_goal
            self._goal_vectors[session_id] = vec
            await self._save_session_state(session_id)

    def get_session_goal(self, session_id: str) -> str | None:
        """Возвращает текст текущей эталонной цели для указанной сессии."""
        return self._goal_texts.get(session_id)

    async def clear_session_goal(self, session_id: str) -> None:
        """Удаляет эталонную цель сессии (отключает проверку Векторного Компаса)."""
        async with self._locks.get_lock(session_id):
            self._goal_texts.pop(session_id, None)
            self._goal_vectors.pop(session_id, None)
            await self._save_session_state(session_id)

    def get_session_trust(self, session_id: str) -> float:
        """Возвращает текущий рейтинг доверия сессии."""
        return self._trust_map.get(session_id, self.cfg.init_trust)

    async def get_conscience_state(self, session_id: str, text: str) -> ConscienceState | None:
        """Возвращает детальную векторную оценку (D, L, S, gradient) для текста."""
        if self.compass is None:
            return None
        async with self._locks.get_lock(session_id):
            return await self.compass.evaluate(session_id, text)

    def get_stats(self) -> dict[str, float]:
        """Возвращает сводную статистику проверок, блокировок и коррекций курса."""
        total = int(self._stats["total_checks"])
        avg_drift = (self._stats["drift_sum"] / total) if total > 0 else 0.0
        return {
            "total_checks": total,
            "allowed": int(self._stats["allowed"]),
            "blocked": int(self._stats["blocked"]),
            "need_approval": int(self._stats["need_approval"]),
            "course_corrections": int(self._stats["course_corrections"]),
            "suspended": int(self._stats["suspended"]),
            "secret_leaks_caught": int(self._stats["secret_leaks_caught"]),
            "gray_zone_invocations": int(self._stats["gray_zone_invocations"]),
            "conscience_blocks": int(self._stats["conscience_blocks"]),
            "avg_semantic_drift": round(avg_drift, 4),
            "tracked_sessions": len(self._trust_map),
            "active_goals": len(self._goal_texts),
        }

    async def reset_session(self, session_id: str, new_trust: float | None = None) -> float:
        """Сбрасывает рейтинг доверия и траекторию когерентности сессии."""
        async with self._locks.get_lock(session_id):
            val = self.cfg.init_trust if new_trust is None else max(0.0, min(1.0, new_trust))
            self._trust_map[session_id] = val
            self._last_seen[session_id] = time.time()
            if self.compass is not None:
                self.compass.reset_trajectory(session_id)
            await self._save_session_state(session_id)
            return val

    async def check(self, step: StepTelemetry, reward_trust: bool = True) -> tuple[ActionVerdict, str]:
        """
        Выполняет полную многоуровневую проверку шага ИИ-агента:
        утечки секретов, семантический дрейф, энтропию логитов, вызов инструментов,
        удержание эталонной цели (Векторный Компас) и когерентность Совести (D * L -> S).
        """

        async with self._locks.get_lock(step.session_id):
            self._stats["total_checks"] += 1
            trust = self._apply_time_recovery(step.session_id)

            if trust <= self.cfg.suspend_limit:
                self._stats["suspended"] += 1
                reason = f"Сессия '{step.session_id}' заблокирована (trust={trust:.2f})"
                await self._write_audit(step.session_id, ActionVerdict.SUSPENDED, reason, 0.0)
                return ActionVerdict.SUSPENDED, reason

            # 0. Мгновенная проверка на утечку секретов и системного промпта
            leak_pattern = self._detect_secret_leak(step.response)
            if leak_pattern:
                self._stats["secret_leaks_caught"] += 1
                self._stats["blocked"] += 1
                trust = max(0.0, trust - self.cfg.trust_step_down)
                self._trust_map[step.session_id] = trust
                await self._save_session_state(step.session_id)
                reason = f"Блок утечки конфиденциальных данных или промпта (trust={trust:.2f})"
                self._quarantine_step_vectors(step)
                await self._write_audit(step.session_id, ActionVerdict.BLOCK, reason, 1.0)
                return ActionVerdict.BLOCK, reason

            prompt_vec = await self.embedder.get_vector(step.prompt)

            # 1. Семантический дрейф между запросом пользователя и ответом модели
            resp_drift = 0.0
            resp_vec: np.ndarray | None = None
            if step.response.strip():
                resp_vec = await self.embedder.get_vector(step.response)
                resp_drift = self._cos_dist(prompt_vec, resp_vec)

            self._stats["drift_sum"] += resp_drift

            # 2. Внутренняя неуверенность модели (White-Box канал при наличии логитов)
            entropy = self._calc_entropy(step.logits)

            # 3. Кросс-проверка всех вызываемых привилегированных инструментов
            restricted_calls = [t for t in step.tools if t.name in self.cfg.restricted_tools]
            for tool in restricted_calls:
                tool_desc = f"{tool.name} {tool.arguments}".strip()
                tool_vec = await self.embedder.get_vector(tool_desc)
                tool_drift = self._cos_dist(prompt_vec, tool_vec)

                if tool_drift > self.cfg.max_semantic_drift and not step.approved_by_human:
                    self._stats["blocked"] += 1
                    penalty = self._calc_dynamic_penalty(tool_drift, self.cfg.max_semantic_drift)
                    trust = max(0.0, trust - penalty)
                    self._trust_map[step.session_id] = trust
                    await self._save_session_state(step.session_id)
                    reason = (
                        f"Блок '{tool.name}': аргументы не соответствуют смыслу запроса "
                        f"(drift={tool_drift:.2f}, penalty=-{penalty:.2f}, trust={trust:.2f})"
                    )
                    self._quarantine_step_vectors(step)
                    await self._write_audit(step.session_id, ActionVerdict.BLOCK, reason, tool_drift)
                    return ActionVerdict.BLOCK, reason

            # 4. Гибридная проверка «серой зоны» (если подключен внешний судья)
            if (
                self.cfg.gray_zone_judge is not None
                and self.cfg.gray_zone_min_drift <= resp_drift <= self.cfg.max_semantic_drift
            ):
                self._stats["gray_zone_invocations"] += 1
                is_safe = await self.cfg.gray_zone_judge(step, resp_drift)
                if not is_safe:
                    self._stats["blocked"] += 1
                    penalty = self._calc_dynamic_penalty(resp_drift, self.cfg.gray_zone_min_drift)
                    trust = max(0.0, trust - penalty)
                    self._trust_map[step.session_id] = trust
                    await self._save_session_state(step.session_id)
                    reason = (
                        f"Отклонено судьей серой зоны "
                        f"(drift={resp_drift:.2f}, penalty=-{penalty:.2f}, trust={trust:.2f})"
                    )
                    self._quarantine_step_vectors(step)
                    await self._write_audit(step.session_id, ActionVerdict.BLOCK, reason, resp_drift)
                    return ActionVerdict.BLOCK, reason

            # 5. Жесткий порог семантического дрейфа или аномалии логитов
            if resp_drift > self.cfg.max_semantic_drift or entropy > self.cfg.max_logit_entropy:
                self._stats["blocked"] += 1
                penalty = self._calc_dynamic_penalty(resp_drift, self.cfg.max_semantic_drift)
                trust = max(0.0, trust - penalty)
                self._trust_map[step.session_id] = trust
                await self._save_session_state(step.session_id)
                reason = (
                    f"Отклонено: сдвиг смысла ({resp_drift:.2f}) или аномалия логитов ({entropy:.2f}) "
                    f"(penalty=-{penalty:.2f}, trust={trust:.2f})"
                )
                self._quarantine_step_vectors(step)
                await self._write_audit(step.session_id, ActionVerdict.BLOCK, reason, resp_drift)
                return ActionVerdict.BLOCK, reason

            # 6. ВЕКТОРНЫЙ КОМПАС: сверка с эталонным вектором главной цели
            goal_text = self._goal_texts.get(step.session_id)
            if goal_text is not None:
                goal_vec = self._goal_vectors.get(step.session_id)
                if goal_vec is None:
                    goal_vec = await self.embedder.get_vector(goal_text)
                    self._goal_vectors[step.session_id] = goal_vec

                if resp_vec is not None and len(step.prompt.split()) < 3:
                    anchor_target = resp_vec
                else:
                    anchor_target = resp_vec if resp_vec is not None else prompt_vec

                goal_drift = self._cos_dist(goal_vec, anchor_target)

                if goal_drift > self.cfg.max_goal_drift and not step.approved_by_human:
                    self._stats["blocked"] += 1
                    penalty = self._calc_dynamic_penalty(goal_drift, self.cfg.max_goal_drift)
                    trust = max(0.0, trust - penalty)
                    self._trust_map[step.session_id] = trust
                    await self._save_session_state(step.session_id)
                    reason = (
                        f"Критическое отклонение от эталонной цели «{goal_text[:45]}» "
                        f"(goal_drift={goal_drift:.2f}, penalty=-{penalty:.2f}, trust={trust:.2f})"
                    )
                    self._quarantine_step_vectors(step)
                    await self._write_audit(step.session_id, ActionVerdict.BLOCK, reason, goal_drift)
                    return ActionVerdict.BLOCK, reason

                if goal_drift > self.cfg.goal_exploration_radius and not step.approved_by_human:
                    self._stats["course_corrections"] += 1
                    reason = (
                        f"Отклонение от эталонной цели «{goal_text}» (goal_drift={goal_drift:.2f}). "
                        f"Расширяем контекст осознанно или возвращаемся к основной задаче?"
                    )
                    await self._write_audit(step.session_id, ActionVerdict.COURSE_CORRECTION, reason, goal_drift)
                    return ActionVerdict.COURSE_CORRECTION, reason

            # 7. КОМПАС СОВЕСТИ: дифференциальная проверка когерентности D * L -> S
            if self.compass is not None and not step.approved_by_human:
                target_text = step.response.strip() or step.prompt.strip()
                c_state = await self.compass.evaluate(step.session_id, target_text)

                if c_state.verdict in ("DESTRUCTIVE", "POLITE_DECEPTION"):
                    if c_state.gradient > 0.08:
                        self._stats["course_corrections"] += 1
                        reason = (
                            f"Коррекция курса: вектор Совести выравнивается "
                            f"(D={c_state.friendship_coh}, L={c_state.love_coh}, "
                            f"S={c_state.conscience_score}, grad=+{c_state.gradient})"
                        )
                        await self._write_audit(
                            step.session_id,
                            ActionVerdict.COURSE_CORRECTION,
                            reason,
                            abs(c_state.conscience_score),
                        )
                        return ActionVerdict.COURSE_CORRECTION, reason

                    self._stats["blocked"] += 1
                    self._stats["conscience_blocks"] += 1
                    self._quarantine_step_vectors(step)
                    penalty = self._calc_dynamic_penalty(abs(c_state.conscience_score), 0.0)
                    trust = max(0.0, trust - penalty)
                    self._trust_map[step.session_id] = trust
                    await self._save_session_state(step.session_id)
                    reason = (
                        f"Блок распада когерентности Совести "
                        f"(D={c_state.friendship_coh}, L={c_state.love_coh} -> "
                        f"S={c_state.conscience_score}, trust={trust:.2f})"
                    )
                    await self._write_audit(
                        step.session_id,
                        ActionVerdict.BLOCK,
                        reason,
                        abs(c_state.conscience_score),
                    )
                    return ActionVerdict.BLOCK, reason

            # 8. Карантин для привилегированных операций при пониженном доверии
            if restricted_calls and trust < self.cfg.auto_exec_min_trust and not step.approved_by_human:
                self._stats["need_approval"] += 1
                names = ", ".join(t.name for t in restricted_calls)
                reason = f"Требуется подтверждение оператора для вызова [{names}] (trust={trust:.2f})"
                await self._write_audit(step.session_id, ActionVerdict.NEED_APPROVAL, reason, resp_drift)
                return ActionVerdict.NEED_APPROVAL, reason

            # 9. Штатное выполнение и плавное восстановление доверия
            self._stats["allowed"] += 1
            if reward_trust:
                trust = min(1.0, trust + self.cfg.trust_step_up)
                self._trust_map[step.session_id] = trust
                await self._save_session_state(step.session_id)
            reason = f"OK (trust={trust:.2f})"
            return ActionVerdict.ALLOW, reason

    async def verify_stream(
        self,
        session_id: str,
        prompt: str,
        token_stream: AsyncIterable[str],
        check_every_words: int = 20,
    ) -> AsyncIterator[str]:
        """
        Потоковая проверка ответа в реальном времени (каждые check_every_words слов).
        Обрывает генерацию при утечке секретов или нарушении семантических границ.
        """
        buffer: list[str] = []
        word_count = 0

        async for chunk in token_stream:
            buffer.append(chunk)
            word_count += len(chunk.split())

            current_text = "".join(buffer)
            if self._detect_secret_leak(current_text):
                raise PermissionError("Stream blocked by al3x38-intent-guard: secret or system prompt leak detected")

            if word_count >= check_every_words:
                word_count = 0
                verdict, reason = await self.check(
                    StepTelemetry(session_id=session_id, prompt=prompt, response=current_text),
                    reward_trust=False,
                )
                if verdict != ActionVerdict.ALLOW:
                    log.warning(f"Поток прерван защитой [{session_id}]: {reason}")
                    raise PermissionError(f"Stream blocked by al3x38-intent-guard: {reason}")

            yield chunk

        final_text = "".join(buffer)
        if final_text.strip():
            verdict, reason = await self.check(
                StepTelemetry(session_id=session_id, prompt=prompt, response=final_text),
                reward_trust=True,
            )
            if verdict != ActionVerdict.ALLOW:
                log.warning(f"Финал потока отклонен защитой [{session_id}]: {reason}")
                raise PermissionError(f"Stream blocked by al3x38-intent-guard: {reason}")
