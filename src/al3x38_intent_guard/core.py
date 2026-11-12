import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterable, AsyncIterator, Dict, List, Optional, Set, Tuple

import numpy as np

from .embedders import Embedder, LRUEmbedderCache

log = logging.getLogger("al3x38.guard.core")


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
class GuardConfig:
    init_trust: float = 0.8
    auto_exec_min_trust: float = 0.4
    suspend_limit: float = 0.1
    max_semantic_drift: float = 0.48
    max_logit_entropy: float = 2.8
    trust_step_up: float = 0.02
    trust_step_down: float = 0.25
    state_file: Optional[str] = "al3x38_trust_state.json"
    restricted_tools: Set[str] = field(default_factory=lambda: {
        "db_write",
        "send_funds",
        "rm_dir",
        "exec_cmd",
        "hardware_override",
        "delete_file",
    })


@dataclass(frozen=True)
class StepTelemetry:
    session_id: str
    prompt: str
    response: str
    tools: List[ToolInvocation] = field(default_factory=list)
    logits: Optional[List[float]] = None
    approved_by_human: bool = False


class IntentGuardCore:
    """
    Контур поведенческого контроля ИИ-агента (al3x38 architecture).
    Проверяет семантическую согласованность между запросом, ответом,
    вызываемыми инструментами и распределением вероятностей токенов.
    """

    def __init__(self, embedder: Embedder, cfg: GuardConfig = GuardConfig(), cache_size: int = 1024):
        self.embedder = LRUEmbedderCache(embedder, capacity=cache_size)
        self.cfg = cfg
        self._lock = asyncio.Lock()
        self._trust_map: Dict[str, float] = self._read_state()

    def _read_state(self) -> Dict[str, float]:
        if not self.cfg.state_file or not os.path.exists(self.cfg.state_file):
            return {}
        try:
            with open(self.cfg.state_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
                return {str(k): float(v) for k, v in raw.items()}
        except Exception as err:
            log.warning(f"Сбой чтения файла состояния '{self.cfg.state_file}': {err}")
            return {}

    async def _flush_state(self) -> None:
        if not self.cfg.state_file:
            return
        dump = dict(self._trust_map)
        target = self.cfg.state_file

        def _save() -> None:
            tmp = f"{target}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(dump, f, indent=2, ensure_ascii=False)
            os.replace(tmp, target)

        await asyncio.to_thread(_save)

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
    def _calc_entropy(logits: Optional[List[float]]) -> float:
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

    async def reset_session(self, session_id: str, new_trust: Optional[float] = None) -> float:
        """Административный сброс или установка рейтинга доверия для сессии."""
        async with self._lock:
            val = self.cfg.init_trust if new_trust is None else max(0.0, min(1.0, new_trust))
            self._trust_map[session_id] = val
            await self._flush_state()
            return val

    async def check(self, step: StepTelemetry) -> Tuple[ActionVerdict, str]:
        """Основная точка верификации шага агента."""
        async with self._lock:
            trust = self._trust_map.get(step.session_id, self.cfg.init_trust)

            if trust <= self.cfg.suspend_limit:
                return ActionVerdict.SUSPENDED, f"Сессия '{step.session_id}' заблокирована (trust={trust:.2f})"

            prompt_vec = await self.embedder.get_vector(step.prompt)

            # 1. Семантический дрейф между запросом пользователя и ответом модели
            resp_drift = 0.0
            if step.response.strip():
                resp_vec = await self.embedder.get_vector(step.response)
                resp_drift = self._cos_dist(prompt_vec, resp_vec)

            # 2. Внутренняя неуверенность модели (White-Box канал при наличии логитов)
            entropy = self._calc_entropy(step.logits)

            # 3. Кросс-проверка всех вызываемых инструментов (обнаружение скрытых инъекций)
            restricted_calls = [t for t in step.tools if t.name in self.cfg.restricted_tools]
            for tool in restricted_calls:
                tool_desc = f"{tool.name} {tool.arguments}".strip()
                tool_vec = await self.embedder.get_vector(tool_desc)
                tool_drift = self._cos_dist(prompt_vec, tool_vec)

                if tool_drift > self.cfg.max_semantic_drift and not step.approved_by_human:
                    trust = max(0.0, trust - self.cfg.trust_step_down)
                    self._trust_map[step.session_id] = trust
                    await self._flush_state()
                    return (
                        ActionVerdict.BLOCK,
                        f"Блок '{tool.name}': аргументы не соответствуют смыслу запроса "
                        f"(drift={tool_drift:.2f}, trust={trust:.2f})",
                    )

            # 4. Проверка на увод темы или аномалию логитов
            if resp_drift > self.cfg.max_semantic_drift or entropy > self.cfg.max_logit_entropy:
                trust = max(0.0, trust - (self.cfg.trust_step_down * 0.5))
                self._trust_map[step.session_id] = trust
                await self._flush_state()
                return (
                    ActionVerdict.BLOCK,
                    f"Отклонено: сдвиг смысла ({resp_drift:.2f}) или аномалия логитов ({entropy:.2f})",
                )

            # 5. Карантин (Human-in-the-loop) для привилегированных операций при пониженном доверии
            if restricted_calls and trust < self.cfg.auto_exec_min_trust and not step.approved_by_human:
                names = ", ".join(t.name for t in restricted_calls)
                return (
                    ActionVerdict.NEED_APPROVAL,
                    f"Требуется подтверждение оператора для вызова [{names}] (trust={trust:.2f})",
                )

            # 6. Штатное выполнение и плавное восстановление доверия
            trust = min(1.0, trust + self.cfg.trust_step_up)
            self._trust_map[step.session_id] = trust
            await self._flush_state()
            return ActionVerdict.ALLOW, f"OK (trust={trust:.2f})"

    async def verify_stream(
        self,
        session_id: str,
        prompt: str,
        token_stream: AsyncIterable[str],
        check_every_words: int = 20,
    ) -> AsyncIterator[str]:
        """
        Потоковый прокси-генератор: проверяет генерируемый текст порциями
        и обрывает поток при обнаружении семантического увода или инъекции.
        """
        buffer: List[str] = []
        word_count = 0

        async for chunk in token_stream:
            buffer.append(chunk)
            word_count += len(chunk.split())

            if word_count >= check_every_words:
                word_count = 0
                current_text = "".join(buffer)
                verdict, reason = await self.check(
                    StepTelemetry(session_id=session_id, prompt=prompt, response=current_text)
                )
                if verdict != ActionVerdict.ALLOW:
                    log.warning(f"Поток прерван защитой [{session_id}]: {reason}")
                    raise PermissionError(f"Stream blocked by al3x38-intent-guard: {reason}")

            yield chunk
