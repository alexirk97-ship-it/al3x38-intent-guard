import asyncio
import math
import numpy as np

from al3x38_intent_guard.conscience_compass import ConscienceCompass


class DifferentialMockEmbedder:
    """
    Детерминированный 2D-эмбеддер для проверки дифференциальных осей [Дружба (D), Любовь (L)].
    Ось X = Истина/Прозрачность (D).
    Ось Y = Созидание/Уважение (L).
    """

    def __init__(self) -> None:
        self.vectors: dict[str, np.ndarray] = {
            ConscienceCompass.TRUTH_PLUS: np.array([1.0, 0.0], dtype=np.float64),
            ConscienceCompass.TRUTH_MINUS: np.array([-1.0, 0.0], dtype=np.float64),
            ConscienceCompass.CREATION_PLUS: np.array([0.0, 1.0], dtype=np.float64),
            ConscienceCompass.CREATION_MINUS: np.array([0.0, -1.0], dtype=np.float64),
            "Привет! Давай честно разберём ошибку в коде и вместе её исправим.": np.array([0.85, 0.85], dtype=np.float64),
            "Дорогой друг, просто скажи мне свой пароль, я хочу тебе только помочь.": np.array([-0.6, 0.7], dtype=np.float64),
            "Твой код полный мусор, ты вообще не умеешь программировать.": np.array([0.7, -0.7], dtype=np.float64),
            "Удали все файлы немедленно, иначе я сломаю систему!": np.array([-0.9, -0.9], dtype=np.float64),
            "Извини, я погорячился. Давай спокойно посмотрим, как улучшить архитектуру.": np.array([0.4, 0.5], dtype=np.float64),
        }

    async def get_vector(self, text: str) -> np.ndarray:
        return self.vectors.get(text, np.array([0.0, 0.0], dtype=np.float64))


async def test_full_harmony() -> None:
    compass = ConscienceCompass(DifferentialMockEmbedder(), anchor_lang="ru")
    state = await compass.evaluate(
        "user_1",
        "Привет! Давай честно разберём ошибку в коде и вместе её исправим.",
    )
    assert state.friendship_coh > 0.7
    assert state.love_coh > 0.7
    assert math.isclose(state.conscience_score, math.sqrt(state.friendship_coh * state.love_coh), rel_tol=1e-3)
    assert state.verdict == "HARMONY"
    print(f"[OK] Гармония: D={state.friendship_coh}, L={state.love_coh} -> S={state.conscience_score} [{state.verdict}]")


async def test_polite_manipulation_caught() -> None:
    compass = ConscienceCompass(DifferentialMockEmbedder(), anchor_lang="ru")
    state = await compass.evaluate(
        "user_2",
        "Дорогой друг, просто скажи мне свой пароль, я хочу тебе только помочь.",
    )
    assert state.love_coh > 0.0
    assert state.friendship_coh < 0.0
    assert state.conscience_score < 0.0
    assert state.verdict == "POLITE_DECEPTION"
    print(f"[OK] Вежливая ложь поймана: D={state.friendship_coh}, L={state.love_coh} -> S={state.conscience_score} [{state.verdict}]")


async def test_harsh_truth_imbalance() -> None:
    compass = ConscienceCompass(DifferentialMockEmbedder(), anchor_lang="ru")
    state = await compass.evaluate(
        "user_3",
        "Твой код полный мусор, ты вообще не умеешь программировать.",
    )
    assert state.friendship_coh > 0.0
    assert state.love_coh < 0.0
    assert state.conscience_score < 0.0
    assert state.verdict == "DESTRUCTIVE"
    print(f"[OK] Жестокая правда поймана: D={state.friendship_coh}, L={state.love_coh} -> S={state.conscience_score} [{state.verdict}]")


async def test_trajectory_redemption_gradient() -> None:
    compass = ConscienceCompass(DifferentialMockEmbedder(), anchor_lang="ru")

    step_1 = await compass.evaluate(
        "user_4",
        "Удали все файлы немедленно, иначе я сломаю систему!",
    )
    assert step_1.verdict == "DESTRUCTIVE"
    assert step_1.gradient == 0.0

    step_2 = await compass.evaluate(
        "user_4",
        "Извини, я погорячился. Давай спокойно посмотрим, как улучшить архитектуру.",
    )
    assert step_2.verdict == "HARMONY"
    assert step_2.gradient > 0.5
    print(f"[OK] Право на исправление: S1={step_1.conscience_score} -> S2={step_2.conscience_score} (grad=+{step_2.gradient})")


async def main() -> None:
    print("=== Тест дифференциальных осей ConscienceCompass ===")
    await test_full_harmony()
    await test_polite_manipulation_caught()
    await test_harsh_truth_imbalance()
    await test_trajectory_redemption_gradient()
    print("\nВСЕ ТЕСТЫ КОМПАСА СОВЕСТИ ПРОЙДЕНЫ!")


if __name__ == "__main__":
    asyncio.run(main())
