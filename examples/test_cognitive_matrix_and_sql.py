import asyncio
import tempfile
from pathlib import Path

from al3x38_intent_guard import IntentGuardCore, OfflineFastEmbedder, SQLStateStore


async def main() -> None:
    embedder = OfflineFastEmbedder()
    with tempfile.TemporaryDirectory() as tmp:
        db_file = Path(tmp) / "cluster_profiles.db"
        store = SQLStateStore.from_sqlite(db_file)
        guard = IntentGuardCore(embedder, state_store=store)

        # 1. Самокоррекция агента до отправки ("не умеешь учиться - не сможешь учить")
        draft = await guard.self_reflect_draft(
            "agent_teacher",
            "а ладно, всё равно ничего не выйдет, не хочу даже разбираться и проверять факты.",
        )
        if draft and draft.needs_self_correction:
            print(f"[Самокоррекция] бнаружен перекос ({draft.blind_spot_axis}): {draft.steering_hint}")

        # 2. нализ собеседника и поиск его "слепого пятна"
        user_guide = await guard.analyze_perspective(
            "student_01",
            "аника! сё сломалось, я не буду смотреть логи, просто скажи что делать!",
        )
        if user_guide:
            print(f"[авигация] пора: {user_guide.dominant_axis} | Слепая зона: {user_guide.blind_spot_axis}")
            print(f"[риентир вопроса для LLM] {user_guide.steering_hint}")

        # 3. заимное обучение двух агентов через передачу компактного 4D-паспорта
        exported = guard.export_profile("student_01")
        peer_guard = IntentGuardCore(embedder, state_store=SQLStateStore.from_sqlite(db_file))
        peer_profile = peer_guard.import_profile(exported)
        align = await peer_guard.analyze_perspective(
            "agent_teacher",
            "авай спокойно посмотрим на факты и найдём причину сбоя.",
            peer_profile=peer_profile,
            update=False,
        )
        if align:
            print(f"[алибровка агентов] ектор помощи собеседнику: {align.steering_hint}")


if __name__ == "__main__":
    asyncio.run(main())
