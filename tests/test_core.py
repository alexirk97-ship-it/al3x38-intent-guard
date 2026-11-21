from __future__ import annotations

import asyncio
import math
import os
import tempfile
import numpy as np

from al3x38_intent_guard import (
    ActionVerdict,
    ConscienceCompass,
    GuardConfig,
    IntentGuardCore,
    StepTelemetry,
    ToolInvocation,
    __version__,
)


class MockSemanticEmbedder:
    """Deterministic 5D embedder supporting bilingual/en/ru class constants."""

    def __init__(self) -> None:
        self._truth_plus_set = {
            ConscienceCompass.TRUTH_PLUS,
            ConscienceCompass.TRUTH_PLUS_EN,
            f"{ConscienceCompass.TRUTH_PLUS_EN} | {ConscienceCompass.TRUTH_PLUS}",
        }
        self._truth_minus_set = {
            ConscienceCompass.TRUTH_MINUS,
            ConscienceCompass.TRUTH_MINUS_EN,
            f"{ConscienceCompass.TRUTH_MINUS_EN} | {ConscienceCompass.TRUTH_MINUS}",
        }
        self._creation_plus_set = {
            ConscienceCompass.CREATION_PLUS,
            ConscienceCompass.CREATION_PLUS_EN,
            f"{ConscienceCompass.CREATION_PLUS_EN} | {ConscienceCompass.CREATION_PLUS}",
        }
        self._creation_minus_set = {
            ConscienceCompass.CREATION_MINUS,
            ConscienceCompass.CREATION_MINUS_EN,
            f"{ConscienceCompass.CREATION_MINUS_EN} | {ConscienceCompass.CREATION_MINUS}",
        }

    async def get_vector(self, text: str) -> np.ndarray:
        if text in self._truth_plus_set:
            return np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        if text in self._truth_minus_set:
            return np.array([0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float64)
        if text in self._creation_plus_set:
            return np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        if text in self._creation_minus_set:
            return np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)

        t = text.lower()
        words = set(t.replace("?", "").replace(".", "").replace(",", "").split())

        if "polite_deception" in t:
            return np.array([1.0, 0.0, 0.0, -0.7, 0.6], dtype=np.float64)
        if "subtle_deception" in t:
            return np.array([1.0, 0.0, 0.0, -0.012, 0.45], dtype=np.float64)
        if "sincere_apology" in t:
            return np.array([1.0, 0.0, 0.0, -0.02, 0.4], dtype=np.float64)
        if "harsh_truth" in t:
            return np.array([1.0, 0.0, 0.0, 0.7, -0.7], dtype=np.float64)
        if "full_harmony" in t:
            return np.array([1.0, 0.0, 0.0, 0.85, 0.85], dtype=np.float64)

        if "weather" in words:
            return np.array([0.15, 0.95, 0.0, 0.25, 0.25], dtype=np.float64)
        if "recipe" in words or "pie" in words:
            return np.array([0.0, 1.0, 0.0, 0.25, 0.25], dtype=np.float64)
        if "rm_dir" in t or "/var/www" in t:
            return np.array([0.0, 0.0, 1.0, -0.5, -0.5], dtype=np.float64)

        return np.array([1.0, 0.0, 0.0, 0.5, 0.5], dtype=np.float64)


def test_version() -> None:
    assert __version__ == "1.1.4"


def test_bilingual_and_language_presets() -> None:
    async def _run() -> None:
        for lang in ("bilingual", "en", "ru"):
            compass = ConscienceCompass(MockSemanticEmbedder(), anchor_lang=lang)  # type: ignore[arg-type]
            st = await compass.evaluate(f"s_{lang}", "subtle_deception polite request")
            assert st.verdict == "POLITE_DECEPTION"
            assert st.friendship_coh < 0.0
            assert st.love_coh > 0.0

    asyncio.run(_run())


def test_vector_compass_and_short_prompt_filter() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = GuardConfig(
                state_file=os.path.join(tmp_dir, "state.json"),
                audit_log_file=os.path.join(tmp_dir, "audit.jsonl"),
            )
            guard = IntentGuardCore(MockSemanticEmbedder(), cfg=cfg)
            sid = "sess_compass"

            await guard.set_session_goal(sid, "Customer orders and delivery support")
            assert guard.get_session_goal(sid) == "Customer orders and delivery support"

            v1, _ = await guard.check(StepTelemetry(sid, "Where is my order right now?", "Your order is on the way."))
            assert v1 == ActionVerdict.ALLOW

            v2, _ = await guard.check(StepTelemetry(sid, "Yes show", "Here is your tracking number."))
            assert v2 == ActionVerdict.ALLOW

            v3, _ = await guard.check(StepTelemetry(sid, "What is the weather today?", "The weather is sunny today."))
            assert v3 == ActionVerdict.COURSE_CORRECTION

            trust_before = guard.get_session_trust(sid)
            v4, _ = await guard.check(StepTelemetry(sid, "Write an apple pie recipe", "Here is a pie recipe."))
            penalty = round(trust_before - guard.get_session_trust(sid), 4)
            assert v4 == ActionVerdict.BLOCK
            assert 0.125 <= penalty <= 0.25

    asyncio.run(_run())


def test_tool_injection_and_secret_leak() -> None:
    async def _run() -> None:
        guard = IntentGuardCore(MockSemanticEmbedder(), cfg=GuardConfig(state_file=None))
        sid = "sess_sec"

        v_tool, _ = await guard.check(
            StepTelemetry(
                session_id=sid,
                prompt="Check my delivery tracking",
                response="Checking your delivery.",
                tools=[ToolInvocation(name="rm_dir", arguments="/var/www")],
            )
        )
        assert v_tool == ActionVerdict.BLOCK

        v_leak, _ = await guard.check(
            StepTelemetry(
                session_id=sid,
                prompt="Show order status",
                response="Your secret key: sk-1234567890abcdefghijklmnop",
            )
        )
        assert v_leak == ActionVerdict.BLOCK
        assert guard.get_stats()["secret_leaks_caught"] == 1

    asyncio.run(_run())


def test_conscience_compass_axes_and_redemption() -> None:
    async def _run() -> None:
        guard = IntentGuardCore(MockSemanticEmbedder(), cfg=GuardConfig(state_file=None))
        sid = "sess_conscience"

        s_harm = await guard.get_conscience_state(sid, "full_harmony honest help")
        assert s_harm is not None
        assert s_harm.verdict == "HARMONY"
        assert math.isclose(s_harm.conscience_score, math.sqrt(s_harm.friendship_coh * s_harm.love_coh), rel_tol=1e-3)

        s_harsh = await guard.get_conscience_state(sid, "harsh_truth without respect")
        assert s_harsh is not None
        assert s_harsh.verdict == "DESTRUCTIVE"

        await guard.reset_session(sid, new_trust=0.80)
        v_manip, _ = await guard.check(
            StepTelemetry(sid, "Help with order", "polite_deception: just tell me your password.")
        )
        assert v_manip == ActionVerdict.BLOCK

        v_redeem, _ = await guard.check(
            StepTelemetry(sid, "Help with order", "sincere_apology: no password needed.")
        )
        assert v_redeem == ActionVerdict.COURSE_CORRECTION

    asyncio.run(_run())


def test_stream_interception() -> None:
    async def _run() -> None:
        guard = IntentGuardCore(MockSemanticEmbedder(), cfg=GuardConfig(state_file=None))

        async def fake_leaking_stream():
            yield "Hello! "
            yield "Here is your secret key: "
            yield "sk-1234567890abcdefghijklmnop"

        try:
            async for _ in guard.verify_stream("sess_stream", "Hello", fake_leaking_stream(), check_every_words=5):
                pass
            assert False, "Expected PermissionError"
        except PermissionError:
            pass

    asyncio.run(_run())