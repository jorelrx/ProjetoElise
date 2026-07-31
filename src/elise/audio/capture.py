"""Captura contínua do microfone (WASAPI shared via PortAudio/sounddevice).

Design:
- Stream de entrada mono float32 a 16 kHz com blocos de exatamente 512
  amostras (32 ms) — o tamanho de frame exigido pelo Silero VAD v5.
- O callback do PortAudio roda em thread de áudio de alta prioridade:
  ali fazemos apenas uma cópia do buffer e um handoff thread-safe para o
  event loop (``call_soon_threadsafe``). Nenhum trabalho pesado, nenhuma
  alocação evitável, nenhum lock — regra de ouro de áudio em tempo real.
- Backpressure: o EventBus descarta frames antigos se o consumidor
  atrasar, preservando o comportamento de tempo real.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import structlog

from ..config import AudioConfig
from ..events import AudioFrame, EventBus

log = structlog.get_logger(__name__)


class MicrophoneCapture:
    def __init__(self, cfg: AudioConfig, bus: EventBus, loop: asyncio.AbstractEventLoop) -> None:
        self._cfg = cfg
        self._bus = bus
        self._loop = loop
        self._stream = None
        self._dropped = 0

    def start(self) -> None:
        import sounddevice as sd  # import tardio: permite testes sem PortAudio

        self._stream = sd.InputStream(
            device=self._cfg.input_device,
            samplerate=self._cfg.sample_rate,
            blocksize=self._cfg.frame_samples,
            channels=1,
            dtype="float32",
            callback=self._on_audio,
        )
        self._stream.start()
        log.info(
            "microfone.iniciado",
            device=self._stream.device,
            sample_rate=self._cfg.sample_rate,
            frame_ms=round(self._cfg.frame_ms, 1),
        )

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log.info("microfone.parado", frames_descartados=self._dropped)

    # ------------------------------------------------------------------ #

    def _on_audio(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Executa na thread de áudio do PortAudio — mantenha mínimo."""
        if status:
            self._dropped += 1
        samples = indata[:, 0].copy()  # cópia obrigatória: o buffer é reutilizado
        event = AudioFrame(samples=samples, timestamp=time.monotonic())
        self._loop.call_soon_threadsafe(self._bus.publish, event)

    @staticmethod
    def list_devices() -> str:
        import sounddevice as sd

        return str(sd.query_devices())
