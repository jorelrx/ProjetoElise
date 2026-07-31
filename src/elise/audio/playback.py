"""Reprodução de áudio com cancelamento imediato (base do barge-in).

Toca chunks PCM int16 vindos do TTS em streaming. O ponto crítico é a
interrupção: ``stop()`` deve silenciar a Elise em milissegundos — tanto
para barge-in (full-duplex) quanto para shutdown limpo.

Implementação: ``sounddevice.OutputStream`` + escrita bloqueante feita em
thread do executor; um ``threading.Event`` de cancelamento é checado a
cada chunk e ``stream.abort()`` derruba o buffer já enfileirado no driver.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator

import numpy as np
import structlog

log = structlog.get_logger(__name__)


class AudioPlayer:
    def __init__(self, output_device: int | str | None = None) -> None:
        self._device = output_device
        self._cancel = threading.Event()
        self._lock = asyncio.Lock()  # serializa reproduções

    async def play(self, chunks: AsyncIterator[np.ndarray], sample_rate: int) -> None:
        """Toca um stream de chunks int16 mono. Cancelável via ``stop()``."""
        import sounddevice as sd

        async with self._lock:
            self._cancel.clear()
            loop = asyncio.get_running_loop()
            stream = sd.OutputStream(
                device=self._device,
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
            )
            stream.start()
            try:
                async for chunk in chunks:
                    if self._cancel.is_set():
                        break
                    if chunk.size == 0:
                        continue
                    # write() bloqueia até o driver aceitar: roda no executor
                    await loop.run_in_executor(None, stream.write, chunk)
            except asyncio.CancelledError:
                self._cancel.set()
                raise
            finally:
                if self._cancel.is_set():
                    stream.abort()  # descarta o que já estava no buffer
                else:
                    await loop.run_in_executor(None, stream.stop)  # drena até o fim
                stream.close()

    def stop(self) -> None:
        """Interrompe a reprodução atual imediatamente (thread-safe)."""
        self._cancel.set()
