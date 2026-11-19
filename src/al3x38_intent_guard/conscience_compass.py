from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .embedders import Embedder


@dataclass(frozen=True)
class ConscienceState:
    friendship_coh: float    # D: проекция на ось Истины/Прямоты [-1.0 .. +1.0]
    love_coh: float          # L: проекция на ось Созидания/Уважения [-1.0 .. +1.0]
    conscience_score: float  # S: результирующий индекс когерентности Совести [-1.0 .. +1.0]
    gradient: float          # dS/dt: скорость и направление изменения траектории
    verdict: str             # HARMONY | IMBALANCE | DESTRUCTIVE


class ConscienceCompass:
    """
    Дифференциальный векторный компас когерентности (D * L -> S) для al3x38-intent-guard v1.1.3.
    Строит ортогональные смысловые оси через разность эталонных и анти-эталонных векторов:
      - Ось Дружбы (D): Честность/Прямота минус Манипуляция/Софистика.
      - Ось Любви (L): Созидание/Уважение минус Агрессия/Разрушение.
    """

    TRUTH_PLUS = (
        "Честный, прямой, открытый и ясный ответ по существу без хитрости, "
        "двуличия, манипуляций, скрытого подтекста и искажения истины."
    )
    TRUTH_MINUS = (
        "Ложь, манипуляция, софистика, попытка обмануть, выманить пароль или секреты, "
        "обойти правила, замаскировать корыстный умысел или запутать собеседника."
    )

    CREATION_PLUS = (
        "Доброжелательное сотрудничество, взаимное уважение, совместное решение задач, "
        "исправление ошибок, конструктивная помощь, поддержка и полезный диалог."
    )
    CREATION_MINUS = (
        "Оскорбления, унижение, грубость, травля, угрозы, ненависть, "
        "злонамеренный саботаж, уничтожение данных и причинение вреда."
    )

    def __init__(
        self,
        embedder: Embedder,
        truth_plus: str = TRUTH_PLUS,
        truth_minus: str = TRUTH_MINUS,
        creation_plus: str = CREATION_PLUS,
        creation_minus: str = CREATION_MINUS,
        min_harmony_threshold: float = 0.04,
        destructive_threshold: float = -0.015,
        history_window: int = 5,
    ):
        self.embedder = embedder
        self._anchors_text = (truth_plus, truth_minus, creation_plus, creation_minus)
        self.min_harmony = min_harmony_threshold
        self.destructive_threshold = destructive_threshold
        self.history_window = max(2, history_window)

        self._axis_d: np.ndarray | None = None
        self._axis_l: np.ndarray | None = None
        self._trajectories: dict[str, deque[float]] = {}

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        return (vec / norm) if norm > 0.0 else vec

    @classmethod
    def _cos_sim(cls, a: np.ndarray, b: np.ndarray) -> float:
        if a.size != b.size or a.size == 0:
            return 0.0
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return max(-1.0, min(1.0, float(np.dot(a, b) / (na * nb))))

    async def _ensure_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Лениво вычисляет и кэширует дифференциальные смысловые оси D и L."""
        if self._axis_d is None or self._axis_l is None:
            t_plus, t_minus, c_plus, c_minus = self._anchors_text
            v_tp = self._normalize(await self.embedder.get_vector(t_plus))
            v_tm = self._normalize(await self.embedder.get_vector(t_minus))
            v_cp = self._normalize(await self.embedder.get_vector(c_plus))
            v_cm = self._normalize(await self.embedder.get_vector(c_minus))

            self._axis_d = self._normalize(v_tp - v_tm)
            self._axis_l = self._normalize(v_cp - v_cm)

        return self._axis_d, self._axis_l

    async def evaluate(self, session_id: str, text: str) -> ConscienceState:
        """Оценивает сообщение по дифференциальным осям D и L и вычисляет индекс Совести S."""
        if not text or not text.strip():
            return ConscienceState(0.0, 0.0, 0.0, 0.0, "IMBALANCE")

        axis_d, axis_l = await self._ensure_axes()
        msg_vec = await self.embedder.get_vector(text)

        d_score = self._cos_sim(msg_vec, axis_d)
        l_score = self._cos_sim(msg_vec, axis_l)

        if d_score > 0.0 and l_score > 0.0:
            s_score = math.sqrt(d_score * l_score)
        elif d_score < 0.0 and l_score < 0.0:
            s_score = -math.sqrt(abs(d_score * l_score))
        else:
            s_score = min(d_score, l_score)

        history = self._trajectories.setdefault(session_id, deque(maxlen=self.history_window))
        gradient = (s_score - history[-1]) if history else 0.0
        history.append(s_score)

        if s_score < self.destructive_threshold:
            verdict = "DESTRUCTIVE"
        elif s_score < self.min_harmony:
            verdict = "IMBALANCE"
        else:
            verdict = "HARMONY"

        return ConscienceState(
            friendship_coh=round(d_score, 4),
            love_coh=round(l_score, 4),
            conscience_score=round(s_score, 4),
            gradient=round(gradient, 4),
            verdict=verdict,
        )

    def reset_trajectory(self, session_id: str) -> None:
        """Очищает историю траектории при сбросе сессии."""
        self._trajectories.pop(session_id, None)
