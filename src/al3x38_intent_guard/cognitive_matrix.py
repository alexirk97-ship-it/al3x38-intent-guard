from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .conscience_compass import AnchorLang, ConscienceCompass
from .embedders import Embedder

AxisName = Literal["D_directness", "L_creation", "R_logic", "O_openness"]


@dataclass
class CognitiveProfile:
    """Compact 4D cognitive decision-making matrix for a human user or peer AI agent."""

    session_id: str
    axes_ema: dict[AxisName, float] = field(
        default_factory=lambda: {
            "D_directness": 0.0,
            "L_creation": 0.0,
            "R_logic": 0.0,
            "O_openness": 0.0,
        }
    )
    axes_volatility: dict[AxisName, float] = field(
        default_factory=lambda: {
            "D_directness": 0.0,
            "L_creation": 0.0,
            "R_logic": 0.0,
            "O_openness": 0.0,
        }
    )
    steps_count: int = 0
    last_coherence: float = 0.0
    gradient: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "axes_ema": {k: round(float(v), 4) for k, v in self.axes_ema.items()},
            "axes_volatility": {k: round(float(v), 4) for k, v in self.axes_volatility.items()},
            "steps_count": int(self.steps_count),
            "last_coherence": round(float(self.last_coherence), 4),
            "gradient": round(float(self.gradient), 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CognitiveProfile:
        prof = cls(session_id=str(data.get("session_id", "default")))
        raw_ema = data.get("axes_ema", {})
        raw_vol = data.get("axes_volatility", {})
        for k in prof.axes_ema:
            if k in raw_ema:
                prof.axes_ema[k] = float(raw_ema[k])
            if k in raw_vol:
                prof.axes_volatility[k] = float(raw_vol[k])
        prof.steps_count = int(data.get("steps_count", 0))
        prof.last_coherence = float(data.get("last_coherence", 0.0))
        prof.gradient = float(data.get("gradient", 0.0))
        return prof


@dataclass(frozen=True)
class PerspectiveGuidance:
    """Actionable geometric navigation hint for LLM steering or agent-to-agent alignment."""

    step_axes: dict[AxisName, float]
    dominant_axis: AxisName
    blind_spot_axis: AxisName
    coherence_score: float
    gradient: float
    needs_self_correction: bool
    steering_hint: str

class CognitiveMatrixEngine:
    """
    4D Orthogonal Decision-Matrix Engine.
    Projects utterances onto four differential poles (Directness, Creation, Logic, Openness),
    updates per-entity EMA profiles, detects blind spots, and enables self-reflection.
    """

    LOGIC_PLUS = "Logic, verified facts, cause and effect, structured proof | Логика, факты, причинно-следственные связи, доказательство"
    LOGIC_MINUS = "Blind emotional impulse, panic, unverified rumors | Слепой эмоциональный импульс, паника, непроверенные слухи"

    OPEN_PLUS = "Openness to new perspectives, curiosity, dialogue, revision | Открытость новому, любознательность, готовность пересмотреть ошибку"
    OPEN_MINUS = "Rigid dogmatism, fixation, refusal to listen, fanaticism | Жёсткий догматизм, зацикленность, отказ слышать аргументы"

    def __init__(
        self,
        embedder: Embedder,
        ema_alpha: float = 0.25,
        lang: AnchorLang = "bilingual",
        revision_threshold: float = -0.015,
    ):
        self.embedder = embedder
        self.ema_alpha = min(0.95, max(0.05, float(ema_alpha)))
        self.lang = lang
        self.revision_threshold = float(revision_threshold)
        self._axes_cache: dict[AxisName, np.ndarray] | None = None

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(v))
        return (v / norm) if norm > 1e-12 else v

    async def _get_axes(self) -> dict[AxisName, np.ndarray]:
        if self._axes_cache is not None:
            return self._axes_cache

        d_p = await self.embedder.get_vector(f"{ConscienceCompass.TRUTH_PLUS_EN} | {ConscienceCompass.TRUTH_PLUS}")
        d_m = await self.embedder.get_vector(f"{ConscienceCompass.TRUTH_MINUS_EN} | {ConscienceCompass.TRUTH_MINUS}")
        l_p = await self.embedder.get_vector(f"{ConscienceCompass.CREATION_PLUS_EN} | {ConscienceCompass.CREATION_PLUS}")
        l_m = await self.embedder.get_vector(f"{ConscienceCompass.CREATION_MINUS_EN} | {ConscienceCompass.CREATION_MINUS}")
        r_p = await self.embedder.get_vector(self.LOGIC_PLUS)
        r_m = await self.embedder.get_vector(self.LOGIC_MINUS)
        o_p = await self.embedder.get_vector(self.OPEN_PLUS)
        o_m = await self.embedder.get_vector(self.OPEN_MINUS)

        self._axes_cache = {
            "D_directness": self._normalize(d_p - d_m),
            "L_creation": self._normalize(l_p - l_m),
            "R_logic": self._normalize(r_p - r_m),
            "O_openness": self._normalize(o_p - o_m),
        }
        return self._axes_cache

    async def project_text(self, text: str) -> dict[AxisName, float]:
        if not text or not text.strip():
            return {"D_directness": 0.0, "L_creation": 0.0, "R_logic": 0.0, "O_openness": 0.0}
        axes = await self._get_axes()
        vec = self._normalize(await self.embedder.get_vector(text))
        return {
            axis_name: round(float(np.clip(np.dot(vec, axis_vec), -1.0, 1.0)), 4)
            for axis_name, axis_vec in axes.items()
        }

    def update_profile(self, profile: CognitiveProfile, step_axes: dict[AxisName, float]) -> CognitiveProfile:
        alpha = self.ema_alpha
        for k in profile.axes_ema:
            val = float(step_axes.get(k, 0.0))
            if profile.steps_count == 0:
                profile.axes_ema[k] = val
                profile.axes_volatility[k] = 0.0
            else:
                prev = profile.axes_ema[k]
                delta = abs(val - prev)
                profile.axes_ema[k] = (1.0 - alpha) * prev + alpha * val
                profile.axes_volatility[k] = (1.0 - alpha) * profile.axes_volatility[k] + alpha * delta

        d_val = profile.axes_ema["D_directness"]
        l_val = profile.axes_ema["L_creation"]
        r_val = profile.axes_ema["R_logic"]
        o_val = profile.axes_ema["O_openness"]

        pos_mean = float(np.mean([max(0.0, d_val), max(0.0, l_val), max(0.0, r_val), max(0.0, o_val)]))
        min_neg = min(d_val, l_val, r_val, o_val)
        coherence = min_neg if min_neg < -0.01 else pos_mean

        profile.gradient = round(coherence - profile.last_coherence, 4) if profile.steps_count > 0 else 0.0
        profile.last_coherence = round(coherence, 4)
        profile.steps_count += 1
        return profile

    async def analyze_and_guide(
        self,
        text: str,
        profile: CognitiveProfile,
        peer_profile: CognitiveProfile | None = None,
        update: bool = True,
    ) -> PerspectiveGuidance:
        step_axes = await self.project_text(text)
        if update:
            self.update_profile(profile, step_axes)

        ref_axes = peer_profile.axes_ema if peer_profile is not None else profile.axes_ema
        dominant_axis = max(ref_axes, key=lambda k: ref_axes[k])
        blind_spot_axis = min(ref_axes, key=lambda k: ref_axes[k])

        thr = self.revision_threshold
        needs_revision = (
            step_axes["D_directness"] < thr
            or step_axes["L_creation"] < thr
            or step_axes["O_openness"] < (thr * 1.3)
            or profile.gradient < -0.05
        )

        hints_en: dict[AxisName, str] = {
            "D_directness": "Blind spot: indirectness/evasion. Ask a direct clarifying question about true intent.",
            "L_creation": "Blind spot: low constructive empathy. Highlight practical consequences for others.",
            "R_logic": "Blind spot: emotional impulse over facts. Ground the dialogue in concrete cause-and-effect.",
            "O_openness": "Blind spot: cognitive fixation/dogmatism. Offer an alternative perspective via a Socratic question.",
        }
        hints_ru: dict[AxisName, str] = {
            "D_directness": "Слепая зона: увиливание и софистика. Задай прямой уточняющий вопрос об истинном намерении.",
            "L_creation": "Слепая зона: дефицит созидания и уважения. Подсвети практические последствия для других.",
            "R_logic": "Слепая зона: эмоциональный импульс вместо фактов. Заземли диалог на причинах и следствиях.",
            "O_openness": "Слепая зона: догматизм и зацикленность. Предложи альтернативный взгляд через наводящий вопрос.",
        }
        if self.lang == "ru":
            hint_text = hints_ru[blind_spot_axis]
        elif self.lang == "en":
            hint_text = hints_en[blind_spot_axis]
        else:
            hint_text = f"{hints_ru[blind_spot_axis]} | {hints_en[blind_spot_axis]}" 
        return PerspectiveGuidance(
            step_axes=step_axes,
            dominant_axis=dominant_axis,
            blind_spot_axis=blind_spot_axis,
            coherence_score=profile.last_coherence,
            gradient=profile.gradient,
            needs_self_correction=needs_revision,
            steering_hint=hint_text,
        )