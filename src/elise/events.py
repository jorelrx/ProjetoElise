"""Eventos do domínio + barramento assíncrono in-process.

Arquitetura orientada a eventos (recomendação do estudo técnico): cada
serviço (captura, VAD, orquestrador) publica/consome eventos tipados via
um EventBus asyncio. Isso mantém os serviços desacoplados e torna trivial
migrar depois para ZeroMQ/Redis (Fase 2+) ou trocar um serviço por outro
processo (ex.: serviço de áudio em Rust) sem tocar no restante.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np

# --------------------------------------------------------------------------- #
# Eventos
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AudioFrame:
    """Frame mono float32 de 16 kHz vindo do microfone (512 amostras)."""

    samples: np.ndarray
    timestamp: float


@dataclass(slots=True)
class SpeechStarted:
    """VAD detectou início de fala (usado para barge-in em full-duplex)."""

    timestamp: float


@dataclass(slots=True)
class WakeWordDetected:
    """Wake word ("Hey Elise") detectada enquanto o orquestrador estava em IDLE."""

    timestamp: float
    score: float
    model: str


@dataclass(slots=True)
class UtteranceCaptured:
    """Enunciado completo do usuário (fala + pre-roll), pronto para STT."""

    audio: np.ndarray  # float32 mono 16 kHz
    duration_s: float


@dataclass(slots=True)
class TranscriptReady:
    text: str
    stt_latency_s: float


@dataclass(slots=True)
class AssistantStateChanged:
    state: str  # "idle" | "listening" | "thinking" | "speaking"


@dataclass(slots=True)
class Shutdown:
    reason: str = "user"


E = TypeVar("E")


# --------------------------------------------------------------------------- #
# Barramento
# --------------------------------------------------------------------------- #


@dataclass
class EventBus:
    """Pub/sub assíncrono por tipo de evento, com backpressure explícita.

    Cada assinante recebe sua própria fila. Filas de áudio usam
    ``maxsize`` para descartar frames antigos sob pressão (áudio atrasado
    é inútil), em vez de acumular latência silenciosamente.
    """

    _subs: dict[type, list[asyncio.Queue]] = field(default_factory=lambda: defaultdict(list))

    def subscribe(self, event_type: type[E], maxsize: int = 0) -> asyncio.Queue[E]:
        q: asyncio.Queue[E] = asyncio.Queue(maxsize=maxsize)
        self._subs[event_type].append(q)
        return q

    def publish(self, event: Any) -> None:
        for q in self._subs[type(event)]:
            if q.full():
                # Descarta o item mais antigo (drop-oldest): mantém tempo real.
                with contextlib.suppress(asyncio.QueueEmpty):  # corrida benigna
                    q.get_nowait()
            q.put_nowait(event)
