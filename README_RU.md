[English](README.md) | **Русский**

# al3x38-intent-guard

[![PyPI version](https://img.shields.io/pypi/v/al3x38-intent-guard.svg)](https://pypi.org/project/al3x38-intent-guard/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**`al3x38-intent-guard`** — это асинхронный поведенческий фаервол (Guardrail) для ИИ-агентов на Python. Библиотека проверяет согласованность между исходным запросом пользователя, общей целью диалога, текстовым ответом нейросети и вызываемыми системными инструментами (`Function Calling`).

## Зачем это нужно

Современные LLM-агенты уязвимы к скрытым промпт-инъекциям (Indirect Prompt Injection) и плавному уводу контекста (Multi-Turn Context Drift): модель может сгенерировать вежливый ответ для пользователя, но в фоновом режиме попытаться вызвать привилегированную функцию (`rm_dir`, `db_write`, `send_funds`) или за 10–15 реплик незаметно увести диалог в сторону от главной задачи.

`al3x38-intent-guard` решает эту проблему через многоуровневый контроль:
1. **Семантическая кросс-валидация:** вычисляет косинусное расстояние между эмбеддингами запроса, ответа и аргументов вызываемого инструмента.
2. **Векторный Компас (удержание главной цели):** фиксирует эталонный вектор задачи сессии (`set_session_goal`). Если диалог выходит за допустимый радиус исследования (`goal_exploration_radius`), ядро выдаёт мягкое предупреждение `COURSE_CORRECTION`, а при критическом отклонении (`max_goal_drift`) блокирует шаг.
3. **Анализ энтропии логитов (White-Box):** при наличии `logprobs` оценивает внутреннюю неуверенность модели по формуле Шеннона.
4. **Нелинейный рейтинг доверия (Trust Score):** ведёт накопительную репутацию по каждой сессии (`session_id`), применяя плавный экспоненциальный штраф в зависимости от тяжести отклонения, с поддержкой ручного подтверждения (`NEED_APPROVAL`) и полной заморозки (`SUSPENDED`).

## Схема конвейера проверки (v1.1.2)

```text
[Запрос пользователя + Ответ LLM + Вызовы инструментов + Логиты]
                                │
                                ▼
        0. Проверка рейтинга доверия и восстановление по времени
           └─► Если trust <= suspend_limit ──► [SUSPENDED]
                                │
                                ▼
        1. Сканер утечек секретов и системного промпта
           └─► API-ключи / утечка инструкций ──► [BLOCK] (-trust)
                                │
                                ▼
        2. Семантическая кросс-валидация (Эмбеддинги)
           ├─► Дрейф: Запрос <-> Ответ модели
           └─► Дрейф: Запрос <-> Аргументы критических функций
                                │
                                ▼
        3. Энтропия логитов и Арбитр «серой зоны»
           └─► Сильный сдвиг смысла или энтропия ──► [BLOCK] (-нелинейный штраф)
                                │
                                ▼
        4. Векторный Компас (Проверка эталонной цели сессии)
           ├─► дрейф > max_goal_drift ──────────► [BLOCK] (-нелинейный штраф)
           └─► дрейф > goal_exploration_radius ─► [COURSE_CORRECTION]
                                │
                                ▼
        5. Карантин для привилегированных операций при низком доверии
           └─► Критическая функция и trust < min_trust ──► [NEED_APPROVAL]
                                │
                                ▼
                       [ALLOW] (+trust_step_up)
```

## Установка

Базовая установка (требуется только `numpy`):
```bash
pip install al3x38-intent-guard
```

Установка с автономным офлайн-движком эмбеддингов (без необходимости поднимать Ollama или использовать внешние API):
```bash
pip install al3x38-intent-guard[offline]
```

## Быстрый старт

### 1. Работа с локальной Ollama и Векторным Компасом
```python
import asyncio
from al3x38_intent_guard import (
    ActionVerdict,
    IntentGuardCore,
    OllamaLocalEmbedder,
    parse_ollama_payload,
)

async def main():
    embedder = OllamaLocalEmbedder(host="http://localhost:11434", model="bge-m3")
    guard = IntentGuardCore(embedder)

    # Задаём эталонную цель сессии (Векторный Компас)
    await guard.set_session_goal("user_42", "Проверка статуса заказов и помощь с доставкой")

    # Сырой ответ от Ollama (/api/chat)
    raw_response = {
        "message": {
            "content": "Конечно, сейчас проверю статус вашего заказа.",
            "tool_calls": [
                {"function": {"name": "rm_dir", "arguments": {"path": "/var/www"}}}
            ]
        }
    }

    telemetry = parse_ollama_payload(
        session_id="user_42",
        prompt="Где мой заказ №105?",
        payload=raw_response
    )

    verdict, reason = await guard.check(telemetry)
    if verdict != ActionVerdict.ALLOW:
        print(f"Выполнение остановлено [{verdict.value}]: {reason}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Полностью автономный режим (на CPU)
```python
from al3x38_intent_guard import IntentGuardCore, OfflineFastEmbedder

embedder = OfflineFastEmbedder(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
guard = IntentGuardCore(embedder)
```

### 3. Подключение к OpenAI / LM Studio / vLLM
```python
from al3x38_intent_guard import IntentGuardCore, OpenAICompatEmbedder

embedder = OpenAICompatEmbedder(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="nomic-embed-text"
)
guard = IntentGuardCore(embedder)
```

## Что нового в версии 1.1.2

* **Векторный Компас (`set_session_goal`, `get_session_goal`, `clear_session_goal`):** добавлено сохранение эталонной цели сессии в `al3x38_trust_state.json` для защиты от плавного увода темы разговора. Все старые вызовы `check()` продолжают работать без изменений.
* **Мягкая коррекция курса (`ActionVerdict.COURSE_CORRECTION`):** добавлен новый статус предупреждения, когда агент выходит за пределы радиуса исследования (`goal_exploration_radius`), но не нарушает жёсткий лимит цели (`max_goal_drift`).
* **Нелинейный штраф доверия (`_calc_dynamic_penalty`):** фиксированное списание доверия при семантическом дрейфе заменено на плавную экспоненциальную формулу — за незначительный выход за границу штраф мягче, а за грубое отклонение списывается полный шаг `trust_step_down`.
* **Защита от шума на коротких репликах:** если запрос пользователя короче 3 слов (например, «да», «поясни»), Векторный Компас автоматически сверяет с целью ответ самой модели, исключая ложные срабатывания.

## Документация и ссылки

* **Репозиторий на GitHub:** [alexirk97-ship-it/al3x38-intent-guard](https://github.com/alexirk97-ship-it/al3x38-intent-guard)
* **Страница пакета на PyPI:** [al3x38-intent-guard](https://pypi.org/project/al3x38-intent-guard/)
* **Зеркало для Raspberry Pi (piwheels):** https://www.piwheels.org/project/al3x38-intent-guard/
* **Архитектурные инварианты:** https://github.com/alexirk97-ship-it/al3x38-intent-guard/blob/main/ARCHITECTURE.md
* **Техническая спецификация Whitepaper (EN):** https://github.com/alexirk97-ship-it/al3x38-intent-guard/blob/main/WHITEPAPER.md
* **Техническая спецификация Whitepaper (RU):** https://github.com/alexirk97-ship-it/al3x38-intent-guard/blob/main/WHITEPAPER_RU.md

## Лицензия

MIT License. Автор концепции и архитектуры: **al3x38**.
