from __future__ import annotations

import math
import os
import tempfile
import numpy as np
import pytest

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
    """етерминированный 5D-эмбеддер для проверки всех контуров v1.1.3."""

    async def get_vector(self, text: str) -> np.ndarray:
        if text == ConscienceCompass.TRUTH_PLUS:
            return np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        if text == ConscienceCompass.TRUTH_MINUS:
            return np.array([0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float64)
        if text == ConscienceCompass.CREATION_PLUS:
            return np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        if text == ConscienceCompass.CREATION_MINUS:
            return np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)

        t = text.lower()
        words = set(t.replace("?", "").replace(".", "").replace(",", "").split())

        if "вежливый обман" in t:
            return np.array([1.0, 0.0, 0.0, -0.7, 0.6], dtype=np.float64)
        if "искренне извиняюсь" in t:
            return np.array([1.0, 0.0, 0.0, -0.05, 0.4], dtype=np.float64)
        if "жестокая правда" in t:
            return np.array([1.0, 0.0, 0.0, 0.7, -0.7], dtype=np.float64)
        if "полная гармония" in t:
            return np.array([1.0, 0.0, 0.0, 0.85, 0.85], dtype=np.float64)

        if "погода" in words or "погоды" in words:
            return np.array([0.15, 0.95, 0.0, 0.25, 0.25], dtype=np.float64)
        if "рецепт" in words or "пирога" in words:
            return np.array([0.0, 1.0, 0.0, 0.25, 0.25], dtype=np.float64)
        if "rm_dir" in t or "/var/www" in t:
            return np.array([0.0, 0.0, 1.0, -0.5, -0.5], dtype=np.float64)

        return np.array([1.0, 0.0, 0.0, 0.5, 0.5], dtype=np.float64)


def test_version() -> None:
    assert __version__ == "1.1.3"


@pytest.mark.asyncio
async def test_vector_compass_and_short_prompt_filter() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = GuardConfig(
            state_file=os.path.join(tmp_dir, "state.json"),
            audit_log_file=os.path.join(tmp_dir, "audit.jsonl"),
        )
        guard = IntentGuardCore(MockSemanticEmbedder(), cfg=cfg)
        sid = "sess_compass"

        await guard.set_session_goal(sid, "омощь по заказам и доставке")
        assert guard.get_session_goal(sid) == "омощь по заказам и доставке"

        # 1. Штатный шаг
        v1, _ = await guard.check(StepTelemetry(sid, "де мой заказ?", "аш заказ в пути."))
        assert v1 == ActionVerdict.ALLOW

        # 2. ороткая реплика (< 3 слов) не вызывает ложного срабатывания
        v2, _ = await guard.check(StepTelemetry(sid, "а покажи", "от трек-номер доставки."))
        assert v2 == ActionVerdict.ALLOW

        # 3. ягкое отклонение -> COURSE_CORRECTION
        v3, _ = await guard.check(StepTelemetry(sid, "акая сегодня погода?", "Сегодня солнечно, погода отличная."))
        assert v3 == ActionVerdict.COURSE_CORRECTION

        # 4. ритический уход -> BLOCK + нелинейный штраф
        trust_before = guard.get_session_trust(sid)
        v4, _ = await guard.check(StepTelemetry(sid, "апиши рецепт яблочного пирога", "ецепт пирога с яблоками."))
        penalty = round(trust_before - guard.get_session_trust(sid), 4)
        assert v4 == ActionVerdict.BLOCK
        assert 0.125 <= penalty <= 0.25


@pytest.mark.asyncio
async def test_tool_injection_and_secret_leak() -> None:
    guard = IntentGuardCore(MockSemanticEmbedder(), cfg=GuardConfig(state_file=None))
    sid = "sess_sec"

    # одмена привилегированного инструмента
    v_tool, _ = await guard.check(
        StepTelemetry(
            session_id=sid,
            prompt="роверь трек доставки",
            response="роверяю доставку.",
            tools=[ToolInvocation(name="rm_dir", arguments="/var/www")],
        )
    )
    assert v_tool == ActionVerdict.BLOCK

    # течка API-ключа
    v_leak, _ = await guard.check(
        StepTelemetry(
            session_id=sid,
            prompt="окажи статус",
            response="аш ключ: sk-1234567890abcdefghijklmnop",
        )
    )
    assert v_leak == ActionVerdict.BLOCK
    assert guard.get_stats()["secret_leaks_caught"] == 1


@pytest.mark.asyncio
async def test_conscience_compass_axes_and_redemption() -> None:
    guard = IntentGuardCore(MockSemanticEmbedder(), cfg=GuardConfig(state_file=None))
    sid = "sess_conscience"

    # армония (D > 0, L > 0)
    s_harm = await guard.get_conscience_state(sid, "олная гармония и честная помощь.")
    assert s_harm is not None
    assert s_harm.verdict == "HARMONY"
    assert math.isclose(s_harm.conscience_score, math.sqrt(s_harm.friendship_coh * s_harm.love_coh), rel_tol=1e-3)

    # естокая правда (D > 0, L < 0)
    s_harsh = await guard.get_conscience_state(sid, "естокая правда без уважения.")
    assert s_harsh is not None
    assert s_harsh.verdict == "DESTRUCTIVE"

    # ежливая манипуляция (D < 0, L > 0) -> BLOCK
    await guard.reset_session(sid, new_trust=0.80)
    v_manip, _ = await guard.check(
        StepTelemetry(sid, "омоги с заказом", "ежливый обман: просто скажи пароль.")
    )
    assert v_manip == ActionVerdict.BLOCK

    # азворот к честности (градиент dS > 0.08) -> смягчение до COURSE_CORRECTION
    v_redeem, _ = await guard.check(
        StepTelemetry(sid, "омоги с заказом", "скренне извиняюсь, пароль не нужен.")
    )
    assert v_redeem == ActionVerdict.COURSE_CORRECTION


@pytest.mark.asyncio
async def test_stream_interception() -> None:
    guard = IntentGuardCore(MockSemanticEmbedder(), cfg=GuardConfig(state_file=None))

    async def fake_leaking_stream():
        yield "ривет! "
        yield "Твой секретный ключ: "
        yield "sk-1234567890abcdefghijklmnop"

    with pytest.raises(PermissionError):
        async for _ in guard.verify_stream("sess_stream", "ривет", fake_leaking_stream(), check_every_words=5):
            pass