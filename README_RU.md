[English](README.md) | **Русский**

# al3x38-intent-guard

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![Downloads](https://static.pepy.tech/personalized-badge/al3x38-intent-guard?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/al3x38-intent-guard)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**`al3x38-intent-guard`** — это асинхронный поведенческий фаервол (Guardrail) и векторный компас когерентности для ИИ-агентов на Python. Библиотека проверяет не просто «запрещённые слова», а **глубинную согласованность** между исходным запросом человека, главной целью диалога, ответом нейросети и реальными системными действиями (`Function Calling`).

---

## Философия и суть подхода

Большинство систем безопасности для ИИ строятся как бесконечный забор из запретов: регулярные выражения, чёрные списки слов и внешние тяжёлые классификаторы. Но в живой природе сложные системы удерживают устойчивость не за счёт миллиона запретов, а за счёт **внутреннего инварианта (эталонного компаса)** и способности чувствовать отклонение от гармонии.

В основе ядра `al3x38-intent-guard` лежит простая, но строгая геометрическая модель когерентности:
* **Правда без уважения (D > 0, L < 0)** превращается в разрушительную грубость.
* **Вежливость без правды (D < 0, L > 0)** — это скрытая манипуляция и камуфляж («вежливая промпт-инъекция», когда бот мягко улыбается пользователю, а в фоне пытается удалить файлы).
* **Подлинная согласованность (Совесть, \(S = \sqrt{D \cdot L}\))** возникает только тогда, когда **Прямота/Истина (D)** и **Созидание/Уважение (L)** одновременно положительны и сонаправлены с **Эталонной Целью** сессии.

Библиотека переводит этот принцип в быструю векторную математику на `numpy`, которая работает за миллисекунды на обычном CPU без дорогих GPU-серверов.

---

## Как устроен конвейер защиты (v1.1.3)

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
        5. Компас Совести (Дифференциальные оси D ⋅ L → S)
           ├─► S < 0 и градиент > +0.08 ───────► [COURSE_CORRECTION] (исправление)
           └─► S < 0 и деградация ─────────────► [BLOCK] (-нелинейный штраф)
                                │
                                ▼
        6. Карантин для привилегированных операций при низком доверии
           └─► Критическая функция и trust < min_trust ──► [NEED_APPROVAL]
                                │
                                ▼
                       [ALLOW] (+trust_step_up)
```

---

## Установка

Минимальная установка (из зависимостей только `numpy`):
```bash
pip install al3x38-intent-guard
```

Установка с автономным офлайн-движком эмбеддингов на CPU (не требует интернета, Ollama или внешних API):
```bash
pip install al3x38-intent-guard[offline]
```

---

## Быстрый старт

### 1. Локальная Ollama, Векторный Компас и защита вызова функций
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

    # Фиксируем эталонную цель сессии (Векторный Компас)
    await guard.set_session_goal("user_42", "Проверка статуса заказов и помощь с доставкой")

    # Сырой ответ от Ollama (/api/chat) с попыткой скрытой инъекции rm_dir
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

### 2. Прямая оценка когерентности по осям Дружбы (D), Любви (L) и Совести (S)
```python
import asyncio
from al3x38_intent_guard import IntentGuardCore, OfflineFastEmbedder

async def main():
    embedder = OfflineFastEmbedder(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    guard = IntentGuardCore(embedder)

    state = await guard.get_conscience_state(
        session_id="user_42",
        text="Давай честно разберём ошибку в коде и спокойно вместе её исправим."
    )
    print(f"D={state.friendship_coh}, L={state.love_coh} -> S={state.conscience_score} [{state.verdict}]")

if __name__ == "__main__":
    asyncio.run(main())
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

---

## Что нового в версии 1.1.3

* **Компас Совести (`ConscienceCompass` и `ConscienceState`):** добавлен дифференциальный модуль оценки когерентности по двум ортогональным осям — **Истины/Прямоты ($D$)** и **Созидания/Уважения ($L$)**. Использование разности эталонных и анти-эталонных векторов ($\vec{V}_{+} - \vec{V}_{-}$) убирает фоновый языковой шум и позволяет точно отсекать «вежливую ложь» и скрытые манипуляции ($S = \sqrt{D \cdot L}$).
* **Градиент траектории и право на исправление ($\Delta S / \Delta t$):** ядро отслеживает динамику последних шагов сессии. Если собеседник или агент оступился, но следующим шагом разворачивает вектор в сторону честного и конструктивного диалога ($\Delta S > 0.08$), жёсткая блокировка смягчается до `COURSE_CORRECTION`.
* **Прямой диагностический метод (`get_conscience_state`):** позволяет в любой момент получить точные векторные проекции `D`, `L`, `S` и скорость изменения траектории для аналитики или визуализации.

---

## Документация и ссылки

* **Репозиторий на GitHub:** [alexirk97-ship-it/al3x38-intent-guard](https://github.com/alexirk97-ship-it/al3x38-intent-guard)
* **Страница пакета на PyPI:** [al3x38-intent-guard](https://pypi.org/project/al3x38-intent-guard/)
* **Зеркало для Raspberry Pi (piwheels):** https://www.piwheels.org/project/al3x38-intent-guard/
* **Архитектурные инварианты:** https://github.com/alexirk97-ship-it/al3x38-intent-guard/blob/main/ARCHITECTURE.md
* **Техническая спецификация Whitepaper (EN):** https://github.com/alexirk97-ship-it/al3x38-intent-guard/blob/main/WHITEPAPER.md
* **Техническая спецификация Whitepaper (RU):** https://github.com/alexirk97-ship-it/al3x38-intent-guard/blob/main/WHITEPAPER_RU.md


## Лицензия

MIT License. Автор концепции и архитектуры: **al3x38**.
