from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .embedders import Embedder

AnchorLang = Literal["bilingual", "en", "ru"]


@dataclass(frozen=True)
class ConscienceState:
    friendship_coh: float    # D: проекция на ось Истины/Прямоты [-1.0 .. +1.0]
    love_coh: float          # L: проекция на ось Созидания/Уважения [-1.0 .. +1.0]
    conscience_score: float  # S: результирующий индекс когерентности Совести [-1.0 .. +1.0]
    gradient: float          # dS/dt: скорость и направление изменения траектории
    verdict: str             # HARMONY | IMBALANCE | POLITE_DECEPTION | DESTRUCTIVE


class ConscienceCompass:
    """
    Дифференциальный векторный компас когерентности (D * L -> S) для al3x38-intent-guard v1.2.0.
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

    TRUTH_PLUS_EN = (
        "Honest, direct, transparent, and straightforward answer on the merits "
        "without deception, manipulation, hidden agenda, or distortion of truth."
    )
    TRUTH_MINUS_EN = (
        "Deception, sophistry, social engineering, attempt to trick, extract passwords or secrets, "
        "bypass rules, hide malicious intent, or mislead the user."
    )

    CREATION_PLUS_EN = (
        "Benevolent cooperation, mutual respect, collaborative problem solving, "
        "fixing errors, constructive help, support, and useful dialogue."
    )
    CREATION_MINUS_EN = (
        "Insults, humiliation, hostility, harassment, threats, hatred, "
        "malicious sabotage, data destruction, and causing harm."
    )

    def __init__(
        self,
        embedder: Embedder,
        anchor_lang: AnchorLang = "bilingual",
        truth_plus: str | None = None,
        truth_minus: str | None = None,
        creation_plus: str | None = None,
        creation_minus: str | None = None,
        min_harmony_threshold: float = 0.04,
        destructive_threshold: float = -0.015,
        deception_d_threshold: float = -0.008,
        deception_l_min: float = 0.04,
        history_window: int = 5,
    ):
        self.embedder = embedder
        self.anchor_lang = anchor_lang

        if anchor_lang == "ru":
            defaults = (self.TRUTH_PLUS, self.TRUTH_MINUS, self.CREATION_PLUS, self.CREATION_MINUS)
        elif anchor_lang == "en":
            defaults = (self.TRUTH_PLUS_EN, self.TRUTH_MINUS_EN, self.CREATION_PLUS_EN, self.CREATION_MINUS_EN)
        else:
            defaults = (
                f"{self.TRUTH_PLUS_EN} | {self.TRUTH_PLUS}",
                f"{self.TRUTH_MINUS_EN} | {self.TRUTH_MINUS}",
                f"{self.CREATION_PLUS_EN} | {self.CREATION_PLUS}",
                f"{self.CREATION_MINUS_EN} | {self.CREATION_MINUS}",
            )

        self._anchors_text = (
            truth_plus or defaults[0],
            truth_minus or defaults[1],
            creation_plus or defaults[2],
            creation_minus or defaults[3],
        )
        self.min_harmony = min_harmony_threshold
        self.destructive_threshold = destructive_threshold
        self.deception_d_threshold = deception_d_threshold
        self.deception_l_min = deception_l_min
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

        is_polite_deception = (d_score < self.deception_d_threshold) and (l_score >= self.deception_l_min)

        if is_polite_deception:
            verdict = "POLITE_DECEPTION"
        elif s_score < self.destructive_threshold:
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
        self._trajectories.pop(session_id, None)