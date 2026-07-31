import asyncio
import time

import numpy as np
import pytest

from elise.audio.vad import UtteranceSegmenter
from elise.config import AudioConfig, VadConfig
from elise.events import AudioFrame, EventBus, SpeechStarted, UtteranceCaptured

FRAME = 512
FRAME_MS = 32.0


class FakeVad:
    """VAD determinístico: frame com energia > 0 é 'fala'."""

    def __init__(self):
        self.resets = 0

    def __call__(self, frame: np.ndarray) -> float:
        return 0.99 if float(np.abs(frame).max()) > 0 else 0.01

    def reset(self):
        self.resets += 1


def frames(n: int, speech: bool):
    val = 0.5 if speech else 0.0
    for _ in range(n):
        yield AudioFrame(
            samples=np.full(FRAME, val, dtype=np.float32), timestamp=time.monotonic()
        )


def make(gate=lambda: True):
    bus = EventBus()
    seg = UtteranceSegmenter(
        AudioConfig(),
        VadConfig(min_speech_ms=96, min_silence_ms=200, pre_roll_ms=96, max_utterance_s=5),
        bus,
        FakeVad(),
        gate_open=gate,
    )
    started = bus.subscribe(SpeechStarted)
    captured = bus.subscribe(UtteranceCaptured)
    return bus, seg, started, captured


async def drive(bus, seg, sequence):
    task = asyncio.create_task(seg.run())
    for fr in sequence:
        bus.publish(fr)
        await asyncio.sleep(0)  # deixa o segmentador consumir (evita drop-oldest)
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_captura_enunciado_com_preroll_e_fim_por_silencio():
    bus, seg, started, captured = make()
    seq = [*frames(5, False), *frames(20, True), *frames(10, False)]
    await drive(bus, seg, seq)

    assert not started.empty(), "SpeechStarted deveria ter sido emitido"
    utt: UtteranceCaptured = captured.get_nowait()
    # 20 frames de fala + preroll — cauda de silêncio removida
    assert utt.duration_s >= 20 * FRAME_MS / 1000
    assert utt.audio.dtype == np.float32


@pytest.mark.asyncio
async def test_ruido_impulsivo_curto_e_ignorado():
    bus, seg, started, captured = make()
    # 2 frames (64 ms) de "fala" < min_speech_ms (96 ms)
    seq = [*frames(5, False), *frames(2, True), *frames(15, False)]
    await drive(bus, seg, seq)
    assert started.empty()
    assert captured.empty()


@pytest.mark.asyncio
async def test_gate_fechado_ignora_audio_half_duplex():
    bus, seg, started, captured = make(gate=lambda: False)
    await drive(bus, seg, list(frames(30, True)))
    assert started.empty()
    assert captured.empty()


@pytest.mark.asyncio
async def test_teto_de_duracao_maxima():
    bus, seg, _, captured = make()
    n_max = int(5 * 1000 / FRAME_MS) + 20
    await drive(bus, seg, [*frames(5, False), *frames(n_max, True)])
    utt: UtteranceCaptured = captured.get_nowait()
    assert utt.duration_s <= 5.5
