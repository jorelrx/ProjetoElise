"""Wake word "Hey Elise" — detector opcional (openWakeWord) consumindo AudioFrame.

Roda como assinante independente do bus (não dentro do UtteranceSegmenter):
enquanto a Elise está em IDLE, só este detector processa áudio (VAD/STT ficam
parados); ao acordar, é o inverso. Nunca os dois caminhos pesados rodam juntos.

O openWakeWord exige chunks de exatamente 1280 amostras (80 ms @16 kHz) —
``ChunkAccumulator`` faz a ponte com os frames de 512 amostras do projeto.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import structlog

from ..config import WakeWordConfig
from ..events import AudioFrame, EventBus, WakeWordDetected

log = structlog.get_logger(__name__)

CHUNK_SAMPLES = 1280


class WakeWordModel(Protocol):
    def predict(self, chunk: np.ndarray) -> float: ...

    def reset(self) -> None: ...


class ChunkAccumulator:
    """Reagrupa frames de 512 amostras em chunks de 1280 (resíduo entre chamadas)."""

    def __init__(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)

    @property
    def pending_samples(self) -> int:
        return self._buffer.shape[0]

    def push(self, frame: np.ndarray) -> list[np.ndarray]:
        self._buffer = np.concatenate([self._buffer, frame])
        chunks: list[np.ndarray] = []
        while self._buffer.shape[0] >= CHUNK_SAMPLES:
            chunks.append(self._buffer[:CHUNK_SAMPLES])
            self._buffer = self._buffer[CHUNK_SAMPLES:]
        return chunks

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)


class WakeWordDetector:
    """Consome AudioFrame do bus e publica WakeWordDetected ao reconhecer o gatilho.

    ``armed`` controla quando o detector roda de fato (tipicamente: estado IDLE
    do orquestrador) — desarmado, os frames são descartados e o estado interno
    é zerado, mesmo padrão do gate do UtteranceSegmenter.
    """

    def __init__(
        self,
        cfg: WakeWordConfig,
        bus: EventBus,
        model: WakeWordModel,
        armed: Callable[[], bool],
    ) -> None:
        self._cfg = cfg
        self._bus = bus
        self._model = model
        self._armed = armed
        self._queue = bus.subscribe(AudioFrame, maxsize=64)
        self._acc = ChunkAccumulator()
        self._cooldown_until = 0.0

    async def run(self) -> None:
        while True:
            frame: AudioFrame = await self._queue.get()
            if not self._armed():
                if self._acc.pending_samples:
                    self._acc.reset()
                    self._model.reset()
                continue
            for chunk in self._acc.push(frame.samples):
                await self._process_chunk(chunk)

    async def _process_chunk(self, chunk: np.ndarray) -> None:
        now = time.monotonic()
        if now < self._cooldown_until:
            return
        int16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
        loop = asyncio.get_running_loop()
        score = await loop.run_in_executor(None, self._model.predict, int16)
        if score >= self._cfg.threshold:
            log.info("wakeword.detectada", score=round(score, 3), modelo=self._cfg.model)
            self._bus.publish(WakeWordDetected(timestamp=now, score=score, model=self._cfg.model))
            self._model.reset()
            self._cooldown_until = now + self._cfg.cooldown_s


def resolve_model_name(cfg: WakeWordConfig) -> str:
    """Resolve o modelo efetivo: se ``model`` for um .onnx que ainda não existe em
    disco (custom não treinado), cai para ``fallback_model`` (nome pré-treinado)."""
    caminho = Path(cfg.model)
    if caminho.suffix == ".onnx" and not caminho.exists():
        log.info(
            "wakeword.modelo_custom_ausente",
            esperado=str(caminho),
            usando=cfg.fallback_model,
        )
        return cfg.fallback_model
    return cfg.model


class OpenWakeWordModel:
    def __init__(self, cfg: WakeWordConfig) -> None:
        from openwakeword import utils
        from openwakeword.model import Model

        nome = resolve_model_name(cfg)
        # Model() não baixa os .onnx sozinho — é preciso buscá-los primeiro
        # (feature extractors + o modelo em si). Idempotente: só baixa o que
        # ainda não existe em disco.
        utils.download_models(model_names=[nome])
        self._model = Model(wakeword_models=[nome], inference_framework="onnx")
        self._name = list(self._model.models.keys())[0]

    def predict(self, chunk: np.ndarray) -> float:
        result = self._model.predict(chunk)
        return float(result[self._name])

    def reset(self) -> None:
        self._model.reset()


def create_wakeword_model(cfg: WakeWordConfig) -> WakeWordModel | None:
    if not cfg.enabled:
        return None
    try:
        return OpenWakeWordModel(cfg)
    except Exception:  # noqa: BLE001
        log.warning(
            "wakeword.indisponivel",
            dica="pip install elise[wakeword] (openwakeword ausente ou modelo inválido)",
        )
        return None
