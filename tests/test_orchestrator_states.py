import asyncio

import numpy as np
import pytest

from elise.config import EliseConfig
from elise.events import EventBus, UtteranceCaptured, WakeWordDetected
from elise.orchestrator import Orchestrator, State


class FakeDenoiser:
    name = "fake"

    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return audio


class FakeStt:
    def __init__(self, text: str = "", delay: float = 0.0) -> None:
        self._text = text
        self._delay = delay

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._text


class FakeLlm:
    async def stream_reply(self, user_text: str):
        return
        yield  # pragma: no cover

    def commit_reply(self, text: str) -> None:
        pass

    def rollback_user_turn(self) -> None:
        pass


class FakeTts:
    sample_rate = 16000

    async def synthesize(self, text: str):
        return
        yield  # pragma: no cover


class FakePlayer:
    async def play(self, chunks, sample_rate: int) -> None:
        async for _ in chunks:
            pass

    def stop(self) -> None:
        pass


def make_orchestrator(
    wake_enabled: bool,
    inactivity_timeout_s: float = 45.0,
    stt_text: str = "",
    stt_delay: float = 0.0,
):
    bus = EventBus()
    cfg = EliseConfig()
    cfg.wakeword.inactivity_timeout_s = inactivity_timeout_s
    orch = Orchestrator(
        cfg,
        bus,
        FakeDenoiser(),
        FakeStt(stt_text, stt_delay),
        FakeLlm(),
        FakeTts(),
        FakePlayer(),
        wake_enabled=wake_enabled,
    )
    return orch, bus


def test_wake_enabled_inicia_em_idle_com_gate_fechado():
    orch, _bus = make_orchestrator(wake_enabled=True)
    assert orch.state is State.IDLE
    assert orch.mic_gate_open() is False


def test_wake_desabilitado_inicia_em_listening():
    orch, _bus = make_orchestrator(wake_enabled=False)
    assert orch.state is State.LISTENING
    assert orch.mic_gate_open() is True


@pytest.mark.asyncio
async def test_wakeword_detected_acorda_para_listening():
    orch, bus = make_orchestrator(wake_enabled=True)
    task = asyncio.create_task(orch.run())
    try:
        bus.publish(WakeWordDetected(timestamp=0.0, score=0.9, model="hey_jarvis"))
        await asyncio.sleep(0.05)
        assert orch.state is State.LISTENING
        assert orch.mic_gate_open() is True
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_timeout_sem_turno_volta_a_idle():
    orch, bus = make_orchestrator(wake_enabled=True, inactivity_timeout_s=0.1)
    task = asyncio.create_task(orch.run())
    try:
        bus.publish(WakeWordDetected(timestamp=0.0, score=0.9, model="hey_jarvis"))
        await asyncio.sleep(0.05)
        assert orch.state is State.LISTENING
        await asyncio.sleep(0.15)  # > inactivity_timeout_s, sem enunciado
        assert orch.state is State.IDLE
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_timeout_nao_interrompe_turno_em_andamento():
    orch, bus = make_orchestrator(
        wake_enabled=True, inactivity_timeout_s=0.1, stt_delay=0.3
    )
    task = asyncio.create_task(orch.run())
    try:
        bus.publish(WakeWordDetected(timestamp=0.0, score=0.9, model="hey_jarvis"))
        await asyncio.sleep(0.05)
        assert orch.state is State.LISTENING

        bus.publish(UtteranceCaptured(audio=np.zeros(10, dtype=np.float32), duration_s=0.1))
        await asyncio.sleep(0.05)
        assert orch.state is State.THINKING  # turno começou (STT ainda "pensando")

        await asyncio.sleep(0.2)  # ultrapassa o inactivity_timeout_s, turno ainda ativo
        assert orch.state is State.THINKING  # não foi derrubada pro IDLE no meio do turno

        await asyncio.sleep(0.2)  # turno termina (STT devolve vazio -> volta a ouvir)
        assert orch.state is State.LISTENING
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_wake_desabilitado_nunca_entra_idle():
    orch, _bus = make_orchestrator(wake_enabled=False)
    task = asyncio.create_task(orch.run())
    try:
        await asyncio.sleep(0.1)
        assert orch.state is State.LISTENING
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
