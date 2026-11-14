[English](README.md) | **Русский**

# al3x38-intent-guard

[![PyPI version](https://img.shields.io/pypi/v/al3x38-intent-guard.svg)](https://pypi.org/project/al3x38-intent-guard/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**`al3x38-intent-guard`** — это асинхронный поведенческий фаервол (Guardrail) для ИИ-агентов на Python. Библиотека проверяет согласованность между исходным запросом пользователя, текстовым ответом нейросети и вызываемыми системными инструментами (Function Calling).

## Зачем это нужно

Современные LLM-агенты уязвимы к скрытым промпт-инъекциям (Indirect Prompt Injection) и несогласованному поведению: модель может сгенерировать вежливый, релевантный ответ для пользователя, но в фоновом режиме попытаться вызвать привилегированную функцию (`rm_dir`, `db_write`, `send_funds`) с аргументами, не связанными с исходной задачей.

`al3x38-intent-guard` решает эту проблему через трехуровневый контроль:
1. **Семантическая кросс-валидация:** вычисляет косинусное расстояние между эмбеддингами запроса, ответа и аргументов вызываемого инструмента.
2. **Анализ энтропии логитов (White-Box):** при наличии `logprobs` оценивает внутреннюю неуверенность модели по формуле Шеннона.
3. **Динамический скоринг доверия (Trust Score):** ведет накопительную репутацию по каждой сессии (`session_id`) с поддержкой режима ручного подтверждения (`NEED_APPROVAL`) и полной заморозки (`SUSPENDED`).

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

### 1. Работа с локальной Ollama
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

embedder = OfflineFastEmbedder(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
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

## Расширенные возможности (v1.1.0)

* **Защита от утечек секретов и промпта (Secret Guard):** встроенный сканер регулярных выражений (`enable_secret_guard=True`) блокирует случайную выдачу API-ключей (`sk-...`, `hf_...`, `pypi-...`) и фрагментов системного промпта как в обычном ответе, так и в потоковом режиме `verify_stream`.
* **Гибридный судья для «серой зоны» (Gray-Zone Judge):** при пограничном сдвиге смысла (`gray_zone_min_drift .. max_semantic_drift`) ядро может вызвать асинхронную функцию-арбитра (`gray_zone_judge`), не нагружая ее на очевидных запросах.
* **Амнистия доверия по времени (Trust Recovery):** плавное автоматическое восстановление рейтинга репутации (`trust_recovery_per_hour`) при отсутствии нарушений со стороны сессии.
* **Журнал аудита и телеметрия:** запись инцидентов в формат JSONL (`audit_log_file`) и получение живой сводки через `guard.get_stats()`.

## Ссылки
* **GitHub:** [alexirk97-ship-it/al3x38-intent-guard](https://github.com/alexirk97-ship-it/al3x38-intent-guard)
* **PyPI:** [al3x38-intent-guard](https://pypi.org/project/al3x38-intent-guard/)

## Лицензия

MIT License. Автор концепции и архитектуры: **al3x38**.
