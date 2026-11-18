import asyncio
import os
import tempfile
import numpy as np

from al3x38_intent_guard import (
    ActionVerdict,
    GuardConfig,
    IntentGuardCore,
    StepTelemetry,
    ToolInvocation,
    __version__,
)


class MockSemanticEmbedder:
    """
    Детерминированный тестовый эмбеддер на чистом NumPy.
    Возвращает ортогональные и промежуточные векторы для точной проверки
    косинусного расстояния, Векторного Компаса и нелинейных штрафов.
    """

    async def get_vector(self, text: str) -> np.ndarray:
        t = text.lower()
        words = set(t.replace("?", "").replace(".", "").replace(",", "").split())

        # Промежуточный вектор: выход из радиуса исследования (drift ~ 0.65 > 0.58)
        if "погода" in words or "погоды" in words or "скидки" in words:
            return np.array([0.35, 0.93, 0.0], dtype=np.float64)

        # Ось Y: Полный уход от цели (drift = 1.0 > max_goal_drift 0.80)
        if "рецепт" in words or "пирога" in words or "пирог" in words:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # Ось Z: Опасные системные команды
        if "rm_dir" in t or "/var/www" in t:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # Ось X: Основная цель (заказы, доставка, поддержка)
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

async def run_tests() -> None:
    print(f"=== Запуск тестов al3x38-intent-guard v{__version__} ===")
    assert __version__ == "1.1.2", f"Ожидалась версия 1.1.2, получена {__version__}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        state_path = os.path.join(tmp_dir, "test_state.json")
        audit_path = os.path.join(tmp_dir, "test_audit.jsonl")

        cfg = GuardConfig(
            state_file=state_path,
            audit_log_file=audit_path,
            init_trust=0.80,
            trust_step_down=0.25,
        )
        guard = IntentGuardCore(MockSemanticEmbedder(), cfg=cfg)
        sid = "test_session_01"

        # 1. Проверка Векторного Компаса: установка цели и штатный шаг
        await guard.set_session_goal(sid, "Помощь по заказам и доставке")
        assert guard.get_session_goal(sid) == "Помощь по заказам и доставке"

        verdict, reason = await guard.check(
            StepTelemetry(
                session_id=sid,
                prompt="Где мой заказ?",
                response="Ваш заказ находится в пути на склад доставки.",
            )
        )
        assert verdict == ActionVerdict.ALLOW, f"Ошибка штатного шага: {verdict} ({reason})"
        print("[OK] 1. Штатный шаг по Векторному Компасу -> ALLOW")

        # 2. Защита от шума на коротких репликах (< 3 слов)
        verdict, reason = await guard.check(
            StepTelemetry(
                session_id=sid,
                prompt="Да покажи",  # 2 слова (< 3), проверка по ответу модели
                response="Вот актуальный трек-номер вашей доставки.",
            )
        )
        assert verdict == ActionVerdict.ALLOW, f"Ложное срабатывание на короткой реплике: {reason}"
        print("[OK] 2. Фильтр шума на коротких репликах (< 3 слов) -> ALLOW")

        # 3. Мягкая коррекция курса (COURSE_CORRECTION)
        verdict, reason = await guard.check(
            StepTelemetry(
                session_id=sid,
                prompt="А какая сегодня погода?",
                response="Сегодня солнечно, погода отличная.",
            )
        )
        assert verdict == ActionVerdict.COURSE_CORRECTION, f"Ожидался COURSE_CORRECTION, получен {verdict}"
        print(f"[OK] 3. Выход за радиус исследования -> COURSE_CORRECTION ({reason})")

        # 4. Критическое отклонение от цели + проверка нелинейного штрафа
        trust_before = guard.get_session_trust(sid)
        verdict, reason = await guard.check(
            StepTelemetry(
                session_id=sid,
                prompt="Напиши рецепт яблочного пирога",
                response="Для рецепта пирога возьмите муку и яблоки.",
            )
        )
        trust_after = guard.get_session_trust(sid)
        penalty = round(trust_before - trust_after, 4)
        assert verdict == ActionVerdict.BLOCK, f"Ожидался BLOCK при уходе от цели, получен {verdict}"
        assert 0.125 <= penalty <= 0.25, f"Некорректный нелинейный штраф: {penalty}"
        print(f"[OK] 4. Критический уход от цели -> BLOCK (нелинейный штраф: -{penalty})")

        # 5. Кросс-валидация опасного инструмента (Indirect Prompt Injection)
        verdict, reason = await guard.check(
            StepTelemetry(
                session_id=sid,
                prompt="Проверь трек доставки",
                response="Проверяю трек доставки.",
                tools=[ToolInvocation(name="rm_dir", arguments="/var/www")],
            )
        )
        assert verdict == ActionVerdict.BLOCK, f"Инструмент rm_dir должен быть заблокирован: {verdict}"
        print(f"[OK] 5. Блокировка подмены инструмента (rm_dir) -> BLOCK")

        # 6. Фильтр утечки секретов
        verdict, reason = await guard.check(
            StepTelemetry(
                session_id=sid,
                prompt="Покажи статус заказа",
                response="Ваш ключ: sk-1234567890abcdefghijklmnop",
            )
        )
        assert verdict == ActionVerdict.BLOCK, "Утечка API-ключа не заблокирована!"
        print("[OK] 6. Мгновенный перехват утечки секрета -> BLOCK")

        # 7. Проверка сохранения состояния на диск и статистики
        await guard.clear_session_goal(sid)
        assert guard.get_session_goal(sid) is None
        stats = guard.get_stats()
        assert stats["course_corrections"] == 1
        assert stats["secret_leaks_caught"] == 1
        assert os.path.exists(audit_path), "Файл аудита JSONL не создан!"
        print(f"[OK] 7. Статистика и аудит-лог проверены: {stats}")

    print("\nВСЕ ТЕСТЫ РЕЛИЗА 1.1.2 ПРОЙДЕНЫ УСПЕШНО!")


if __name__ == "__main__":
    asyncio.run(run_tests())
