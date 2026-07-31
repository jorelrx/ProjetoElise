"""Speech-to-Text — interface plugável + backend faster-whisper.

faster-whisper (CTranslate2) é a recomendação padrão do estudo: até 4x
mais rápido que o openai/whisper com a mesma acurácia. Na RX 580 o
CTranslate2 não acelera por Vulkan/DirectML, então rodamos CPU int8 com o
modelo ``small`` (pt-BR é idioma Tier-1 do Whisper; WER competitivo).
Alternativa de maior throughput: whisper.cpp com backend Vulkan (troca de
backend futura atrás desta mesma interface).
"""

from __future__ import annotations

import asyncio
from typing import Protocol

import numpy as np
import structlog

from ..config import SttConfig

log = structlog.get_logger(__name__)


class SpeechToText(Protocol):
    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...


class FasterWhisperStt:
    def __init__(self, cfg: SttConfig) -> None:
        from faster_whisper import WhisperModel

        log.info(
            "stt.carregando",
            modelo=cfg.model,
            device=cfg.device,
            compute_type=cfg.compute_type,
        )
        self._cfg = cfg
        self._model = WhisperModel(
            cfg.model, device=cfg.device, compute_type=cfg.compute_type
        )
        # Serializa transcrições: o modelo não é reentrante e a GPU/CPU é uma só.
        self._sem = asyncio.Semaphore(1)

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        assert sample_rate == 16000, "faster-whisper espera 16 kHz"
        async with self._sem:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        segments, _info = self._model.transcribe(
            audio,
            language=self._cfg.language,
            beam_size=self._cfg.beam_size,
            vad_filter=False,  # já segmentamos com Silero no pipeline
            condition_on_previous_text=False,  # evita alucinação em enunciados curtos
        )
        return " ".join(s.text.strip() for s in segments).strip()


def create_stt(cfg: SttConfig) -> SpeechToText:
    if cfg.backend == "faster_whisper":
        return FasterWhisperStt(cfg)
    raise ValueError(f"Backend de STT desconhecido: {cfg.backend}")
