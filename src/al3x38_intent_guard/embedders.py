from __future__ import annotations
import asyncio
import json
import os
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Protocol
import numpy as np


class EmbedderError(RuntimeError):
    """Исключение при сбое получения векторного представления."""


class Embedder(Protocol):
    async def get_vector(self, text: str) -> np.ndarray:
        ...


class LRUEmbedderCache:
    """Потокобезопасный LRU-кэш векторов для снижения нагрузки на API и CPU."""

    def __init__(self, backend: Embedder, capacity: int = 1024):
        self.backend = backend
        self.capacity = max(1, capacity)
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()

    async def get_vector(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(1, dtype=np.float64)

        if text in self._store:
            self._store.move_to_end(text)
            return self._store[text]

        vec = await self.backend.get_vector(text)
        self._store[text] = vec
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)
        return vec


class OpenAICompatEmbedder:
    """Адаптер для OpenAI API, LM Studio, vLLM, LocalAI и совместимых прокси."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "local",
        model: str = "nomic-embed-text",
        timeout: float = 10.0,
    ):
        self.endpoint = f"{base_url.rstrip('/')}/embeddings"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def get_vector(self, text: str) -> np.ndarray:
        def _fetch() -> np.ndarray:
            body = json.dumps({"input": text, "model": self.model}).encode("utf-8")
            req = urllib.request.Request(
                self.endpoint,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as res:
                    raw = json.loads(res.read().decode("utf-8"))
                    items = raw.get("data", [])
                    if not items or "embedding" not in items[0]:
                        raise EmbedderError(f"Некорректный формат ответа от {self.endpoint}")
                    return np.array(items[0]["embedding"], dtype=np.float64)
            except urllib.error.URLError as err:
                raise EmbedderError(f"Ошибка сети при обращении к {self.endpoint}: {err}") from err

        return await asyncio.to_thread(_fetch)


class OllamaLocalEmbedder:
    """Адаптер для локального сервера Ollama (/api/embeddings)."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "bge-m3",
        timeout: float = 10.0,
    ):
        self.endpoint = f"{host.rstrip('/')}/api/embeddings"
        self.model = model
        self.timeout = timeout

    async def get_vector(self, text: str) -> np.ndarray:
        def _fetch() -> np.ndarray:
            body = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
            req = urllib.request.Request(
                self.endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as res:
                    raw = json.loads(res.read().decode("utf-8"))
                    if "embedding" not in raw:
                        raise EmbedderError(f"Ключ 'embedding' отсутствует в ответе Ollama: {raw}")
                    return np.array(raw["embedding"], dtype=np.float64)
            except urllib.error.URLError as err:
                raise EmbedderError(f"Ollama недоступна по адресу {self.endpoint}: {err}") from err

        return await asyncio.to_thread(_fetch)


class OfflineFastEmbedder:
    """Автономный офлайн-драйвер на базе ONNX (требует pip install al3x38-intent-guard[offline])."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        try:
            from fastembed import TextEmbedding
        except ImportError as err:
            raise ImportError(
                "Для автономного режима установите пакет с опцией offline: "
                "pip install al3x38-intent-guard[offline]"
            ) from err

        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"

        kwargs = {"model_name": model_name}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if local_files_only:
            kwargs["local_files_only"] = True

        try:
            self._engine = TextEmbedding(**kwargs)
        except TypeError:
            kwargs.pop("local_files_only", None)
            self._engine = TextEmbedding(**kwargs)

    async def get_vector(self, text: str) -> np.ndarray:
        def _run() -> np.ndarray:
            vec = next(self._engine.embed([text]))
            return np.array(vec, dtype=np.float64)

        return await asyncio.to_thread(_run)
