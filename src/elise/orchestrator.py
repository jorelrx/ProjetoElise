"""Orquestrador — máquina de estados do turno conversacional.

    LISTENING ──UtteranceCaptured──▶ THINKING ──1ª sentença──▶ SPEAKING
        ▲                                                        │
        └────────────── fim da fala / barge-in ──────────────────┘

Responsabilidades:
- Consumir enunciados do VAD e disparar o pipeline denoise → STT → LLM
  (streaming) → segmentação de sentenças → TTS → playback.
- Pipelining: enquanto uma sentença toca, a próxima já está sendo
  sintetizada (fila produtor/consumidor) — latência mínima percebida.
- Half-duplex: fecha o "gate" do microfone enquanto a Elise pensa/fala.
- Full-duplex (experimental): barge-in — SpeechStarted durante SPEAKING
  cancela LLM+TTS+playback instantaneamente.
- Memória honesta: só entra no histórico o que foi efetivamente falado.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from enum import Enum

import numpy as np
import structlog

from .audio.denoise import Denoiser
from .audio.playback import AudioPlayer
from .config import DuplexMode, EliseConfig
from .events import EventBus, SpeechStarted, UtteranceCaptured
from .llm import OpenAICompatChat
from .llm.sentence_stream import sentences
from .stt import SpeechToText
from .tts import TextToSpeech

log = structlog.get_logger(__name__)


class State(str, Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class Orchestrator:
    def __init__(
        self,
        cfg: EliseConfig,
        bus: EventBus,
        denoiser: Denoiser,
        stt: SpeechToText,
        llm: OpenAICompatChat,
        tts: TextToSpeech,
        player: AudioPlayer,
    ) -> None:
        self._cfg = cfg
        self._bus = bus
        self._denoiser = denoiser
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._player = player

        self.state = State.LISTENING
        self._turn_task: asyncio.Task | None = None
        self._utterances = bus.subscribe(UtteranceCaptured, maxsize=4)
        self._speech_started = bus.subscribe(SpeechStarted, maxsize=8)

    # ------------------------------------------------------------------ #
    # Gate do microfone (consultado pelo UtteranceSegmenter)
    # ------------------------------------------------------------------ #

    def mic_gate_open(self) -> bool:
        if self._cfg.behavior.mode is DuplexMode.FULL:
            return True  # sempre ouvindo (barge-in)
        return self.state is State.LISTENING

    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        barge_task = None
        if self._cfg.behavior.mode is DuplexMode.FULL:
            barge_task = asyncio.create_task(self._barge_in_watch(), name="barge-in")
        try:
            while True:
                utt = await self._utterances.get()
                if self._turn_task and not self._turn_task.done():
                    # Novo enunciado durante um turno: interrompe o anterior.
                    await self._cancel_turn()
                self._turn_task = asyncio.create_task(self._run_turn(utt), name="turno")
        finally:
            if barge_task:
                barge_task.cancel()
            await self._cancel_turn()

    async def _barge_in_watch(self) -> None:
        while True:
            await self._speech_started.get()
            if self.state is State.SPEAKING:
                log.info("barge_in.interrompendo")
                self._player.stop()
                await self._cancel_turn()
                self._set_state(State.LISTENING)

    async def _cancel_turn(self) -> None:
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn_task
        self._turn_task = None

    def _set_state(self, state: State) -> None:
        if state is not self.state:
            self.state = state
            log.debug("estado", novo=state.value)

    # ------------------------------------------------------------------ #
    # Um turno completo
    # ------------------------------------------------------------------ #

    async def _run_turn(self, utt: UtteranceCaptured) -> None:
        spoken_parts: list[str] = []
        committed = False
        t0 = time.monotonic()
        try:
            self._set_state(State.THINKING)

            # 1) Denoise do enunciado (CPU-bound → executor)
            loop = asyncio.get_running_loop()
            clean = await loop.run_in_executor(
                None, self._denoiser.enhance, utt.audio, self._cfg.audio.sample_rate
            )

            # 2) STT
            text = await self._stt.transcribe(clean, self._cfg.audio.sample_rate)
            t_stt = time.monotonic() - t0
            if not text or len(text.strip()) < 2:
                log.info("turno.transcricao_vazia", latencia_stt_s=round(t_stt, 2))
                self._set_state(State.LISTENING)
                return
            log.info("usuario", texto=text, latencia_stt_s=round(t_stt, 2))

            # 3) LLM streaming → sentenças → TTS → playback (pipelined)
            token_stream = self._llm.stream_reply(text)
            audio_queue: asyncio.Queue[tuple[str, list[np.ndarray]] | None] = (
                asyncio.Queue(maxsize=2)
            )

            async def producer() -> None:
                """Sintetiza cada sentença assim que ela fecha no stream."""
                try:
                    async for sentence in sentences(token_stream):
                        chunks = [c async for c in self._tts.synthesize(sentence)]
                        await audio_queue.put((sentence, chunks))
                finally:
                    await audio_queue.put(None)

            prod = asyncio.create_task(producer(), name="tts-producer")
            first_audio_at: float | None = None
            try:
                while True:
                    item = await audio_queue.get()
                    if item is None:
                        break
                    sentence, chunks = item
                    if first_audio_at is None:
                        first_audio_at = time.monotonic()
                        log.info(
                            "turno.primeiro_audio",
                            latencia_total_s=round(first_audio_at - t0, 2),
                        )
                        self._set_state(State.SPEAKING)
                    log.info("elise", texto=sentence)
                    await self._player.play(_aiter(chunks), self._tts.sample_rate)
                    spoken_parts.append(sentence)
            finally:
                prod.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await prod

            self._llm.commit_reply(" ".join(spoken_parts))
            committed = True
            log.info("turno.concluido", duracao_s=round(time.monotonic() - t0, 2))
        except asyncio.CancelledError:
            # Barge-in / novo enunciado: registra só o que foi ouvido.
            if spoken_parts:
                self._llm.commit_reply(" ".join(spoken_parts) + " —")
            else:
                self._llm.rollback_user_turn()
            committed = True
            raise
        except Exception:
            log.exception("turno.erro")
            if not committed:
                self._llm.rollback_user_turn()
        finally:
            self._set_state(State.LISTENING)


async def _aiter(chunks: list[np.ndarray]) -> AsyncIterator[np.ndarray]:
    for c in chunks:
        yield c
