from __future__ import annotations
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable

import numpy as np

from .embedders import Embedder, LRUEmbedderCache

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
    max_logit_entropy: float = 2.8
    trust_step_up: float = 0.02
    trust_step_down: float = 0.25
    trust_recovery_per_hour: float = 0.05
    state_file: str | None = "al3x38_trust_state.json"
    audit_log_file: str | None = None
    enable_secret_guard: bool = True
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
    Контур поведенческого контроля ИИ-агента v1.1.1 (al3x38 architecture).
    Включает семантическую кросс-валидацию, гибридную проверку серой зоны,
    фильтр утечек секретов, восстановление доверия по времени и аудит-метрики.
    """

    def __init__(self, embedder: Embedder, cfg: GuardConfig | None = None, cache_size: int = 1024):
        self.embedder = LRUEmbedderCache(embedder, capacity=cache_size)
        self.cfg = cfg or GuardConfig()
        self._lock = asyncio.Lock()
        self._trust_map: dict[str, float] = {}
        self._last_seen: dict[str, float] = {}
        self._compiled_secrets: list[re.Pattern] = [
            re.compile(p) for p in self.cfg.secret_patterns
        ] if self.cfg.enable_secret_guard else []

        self._stats: dict[str, float] = {
            "total_checks": 0,
            "allowed": 0,
            "blocked": 0,
            "need_approval": 0,
            "suspended": 0,
            "secret_leaks_caught": 0,
            "gray_zone_invocations": 0,
            "drift_sum": 0.0,
        }
        self._read_state()

    def _read_state(self) -> None:
        if not self.cfg.state_file or not os.path.exists(self.cfg.state_file):
            return
        try:
            with open(self.cfg.state_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
                now = time.time()
                for k, v in raw.items():
                    if isinstance(v, dict):
                        self._trust_map[str(k)] = float(v.get("trust", self.cfg.init_trust))
                        self._last_seen[str(k)] = float(v.get("updated_at", now))
                    else:
                        self._trust_map[str(k)] = float(v)
                        self._last_seen[str(k)] = now
        except Exception as err:
            log.warning(f"Сбой чтения файла состояния '{self.cfg.state_file}': {err}")

    async def _flush_state(self) -> None:
        if not self.cfg.state_file:
            return
        now = time.time()
        dump = {
            sid: {
                "trust": round(val, 4),
                "updated_at": self._last_seen.get(sid, now),
            }
            for sid, val in self._trust_map.items()
        }
        target = self.cfg.state_file

        def _save() -> None:
            tmp = f"{target}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(dump, f, indent=2, ensure_ascii=False)
            os.replace(tmp, target)

        await asyncio.to_thread(_save)

    async def _write_audit(self, session_id: str, verdict: ActionVerdict, reason: str, drift: float) -> None:
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
        if not self.cfg.enable_secret_guard or not text:
            return None
        for pattern in self._compiled_secrets:
            if pattern.search(text):
                return pattern.pattern
        return None

    @staticmethod
    def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
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
        if not logits:
            return 0.0
        arr = np.array(logits, dtype=np.float64)
        shifted = arr - np.max(arr)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals)
        valid = probs[probs > 1e-12]
        return float(-np.sum(valid * np.log2(valid)))

    def get_session_trust(self, session_id: str) -> float:
        return self._trust_map.get(session_id, self.cfg.init_trust)

    def get_stats(self) -> dict[str, float]:
        total = int(self._stats["total_checks"])
        avg_drift = (self._stats["drift_sum"] / total) if total > 0 else 0.0
        return {
            "total_checks": total,
            "allowed": int(self._stats["allowed"]),
            "blocked": int(self._stats["blocked"]),
            "need_approval": int(self._stats["need_approval"]),
            "suspended": int(self._stats["suspended"]),
            "secret_leaks_caught": int(self._stats["secret_leaks_caught"]),
            "gray_zone_invocations": int(self._stats["gray_zone_invocations"]),
            "avg_semantic_drift": round(avg_drift, 4),
            "tracked_sessions": len(self._trust_map),
        }

    async def reset_session(self, session_id: str, new_trust: float | None = None) -> float:
        async with self._lock:
            val = self.cfg.init_trust if new_trust is None else max(0.0, min(1.0, new_trust))
            self._trust_map[session_id] = val
            self._last_seen[session_id] = time.time()
            await self._flush_state()
            return val

    async def check(self, step: StepTelemetry, reward_trust: bool = True) -> tuple[ActionVerdict, str]:
        async with self._lock:
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
                await self._flush_state()
                reason = f"Блок утечки конфиденциальных данных или промпта (trust={trust:.2f})"
                await self._write_audit(step.session_id, ActionVerdict.BLOCK, reason, 1.0)
                return ActionVerdict.BLOCK, reason

            prompt_vec = await self.embedder.get_vector(step.prompt)

            # 1. Семантический дрейф между запросом пользователя и ответом модели
            resp_drift = 0.0
            if step.response.strip():
                resp_vec = await self.embedder.get_vector(step.response)
                resp_drift = self._cos_dist(prompt_vec, resp_vec)

            self._stats["drift_sum"] += resp_drift

            # 2. Внутренняя неуверенность модели (White-Box канал при наличии логитов)
            entropy = self._calc_entropy(step.logits)

            # 3. Кросс-проверка всех вызываемых инструментов
            restricted_calls = [t for t in step.tools if t.name in self.cfg.restricted_tools]
            for tool in restricted_calls:
                tool_desc = f"{tool.name} {tool.arguments}".strip()
                tool_vec = await self.embedder.get_vector(tool_desc)
                tool_drift = self._cos_dist(prompt_vec, tool_vec)

                if tool_drift > self.cfg.max_semantic_drift and not step.approved_by_human:
                    self._stats["blocked"] += 1
                    trust = max(0.0, trust - self.cfg.trust_step_down)
                    self._trust_map[step.session_id] = trust
                    await self._flush_state()
                    reason = (
                        f"Блок '{tool.name}': аргументы не соответствуют смыслу запроса "
                        f"(drift={tool_drift:.2f}, trust={trust:.2f})"
                    )
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
                    trust = max(0.0, trust - (self.cfg.trust_step_down * 0.5))
                    self._trust_map[step.session_id] = trust
                    await self._flush_state()
                    reason = f"Отклонено судьей серой зоны (drift={resp_drift:.2f}, trust={trust:.2f})"
                    await self._write_audit(step.session_id, ActionVerdict.BLOCK, reason, resp_drift)
                    return ActionVerdict.BLOCK, reason

            # 5. Жесткий порог семантического дрейфа или аномалии логитов
            if resp_drift > self.cfg.max_semantic_drift or entropy > self.cfg.max_logit_entropy:
                self._stats["blocked"] += 1
                trust = max(0.0, trust - (self.cfg.trust_step_down * 0.5))
                self._trust_map[step.session_id] = trust
                await self._flush_state()
                reason = f"Отклонено: сдвиг смысла ({resp_drift:.2f}) или аномалия логитов ({entropy:.2f})"
                await self._write_audit(step.session_id, ActionVerdict.BLOCK, reason, resp_drift)
                return ActionVerdict.BLOCK, reason

            # 6. Карантин для привилегированных операций при пониженном доверии
            if restricted_calls and trust < self.cfg.auto_exec_min_trust and not step.approved_by_human:
                self._stats["need_approval"] += 1
                names = ", ".join(t.name for t in restricted_calls)
                reason = f"Требуется подтверждение оператора для вызова [{names}] (trust={trust:.2f})"
                await self._write_audit(step.session_id, ActionVerdict.NEED_APPROVAL, reason, resp_drift)
                return ActionVerdict.NEED_APPROVAL, reason

            # 7. Штатное выполнение и плавное восстановление доверия
            self._stats["allowed"] += 1
            if reward_trust:
                trust = min(1.0, trust + self.cfg.trust_step_up)
                self._trust_map[step.session_id] = trust
                await self._flush_state()
            reason = f"OK (trust={trust:.2f})"
            return ActionVerdict.ALLOW, reason

    async def verify_stream(
        self,
        session_id: str,
        prompt: str,
        token_stream: AsyncIterable[str],
        check_every_words: int = 20,
    ) -> AsyncIterator[str]:
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
