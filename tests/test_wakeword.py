import asyncio
from pathlib import Path

import numpy as np
import pytest

from elise.audio.wakeword import (
    ChunkAccumulator,
    WakeWordDetector,
    create_wakeword_model,
    resolve_model_name,
)
from elise.config import WakeWordConfig
from elise.events import AudioFrame, EventBus, WakeWordDetected

FRAME = 512


class FakeWakeModel:
    def __init__(self, score: float):
        self.score = score
        self.resets = 0
        self.calls: list[np.ndarray] = []

    def predict(self, chunk: np.ndarray) -> float:
        self.calls.append(chunk)
        return self.score

    def reset(self) -> None:
        self.resets += 1


def frame(val: float = 0.5) -> AudioFrame:
    return AudioFrame(samples=np.full(FRAME, val, dtype=np.float32), timestamp=0.0)


async def drive(bus: EventBus, detector: WakeWordDetector, sequence: list[AudioFrame]) -> None:
    task = asyncio.create_task(detector.run())
    for fr in sequence:
        bus.publish(fr)
        await asyncio.sleep(0)
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# --------------------------------------------------------------------------- #
# ChunkAccumulator
# --------------------------------------------------------------------------- #


def test_chunk_accumulator_agrupa_frames_512_em_chunks_1280():
    acc = ChunkAccumulator()
    frames = [np.arange(FRAME, dtype=np.float32) + i * FRAME for i in range(5)]

    chunks = []
    for fr in frames:
        chunks.extend(acc.push(fr))

    assert len(chunks) == 2
    assert all(c.shape == (1280,) for c in chunks)
    assert np.array_equal(np.concatenate(chunks), np.concatenate(frames)[:2560])
    assert acc.pending_samples == 0


def test_chunk_accumulator_mantem_residuo_entre_chamadas():
    acc = ChunkAccumulator()
    total_chunks = []
    for _ in range(3):
        total_chunks.extend(acc.push(np.zeros(FRAME, dtype=np.float32)))

    assert len(total_chunks) == 1
    assert acc.pending_samples == 256


def test_chunk_accumulator_reset_limpa_residuo():
    acc = ChunkAccumulator()
    acc.push(np.zeros(FRAME, dtype=np.float32))
    acc.reset()
    assert acc.pending_samples == 0


# --------------------------------------------------------------------------- #
# WakeWordDetector
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_publica_wakeword_detected_quando_score_atinge_threshold():
    bus = EventBus()
    model = FakeWakeModel(score=0.9)
    cfg = WakeWordConfig(threshold=0.5, cooldown_s=0)
    detector = WakeWordDetector(cfg, bus, model, armed=lambda: True)
    detected = bus.subscribe(WakeWordDetected)

    await drive(bus, detector, [frame() for _ in range(3)])

    evt: WakeWordDetected = detected.get_nowait()
    assert evt.score == 0.9
    assert evt.model == cfg.model
    assert model.resets == 1  # reset após detecção


@pytest.mark.asyncio
async def test_nao_processa_quando_desarmado():
    bus = EventBus()
    model = FakeWakeModel(score=0.9)
    cfg = WakeWordConfig(threshold=0.5)
    detector = WakeWordDetector(cfg, bus, model, armed=lambda: False)
    detected = bus.subscribe(WakeWordDetected)

    await drive(bus, detector, [frame() for _ in range(5)])

    assert detected.empty()
    assert model.calls == []


@pytest.mark.asyncio
async def test_cooldown_suprime_segunda_deteccao_imediata():
    bus = EventBus()
    model = FakeWakeModel(score=0.9)
    cfg = WakeWordConfig(threshold=0.5, cooldown_s=1000.0)
    detector = WakeWordDetector(cfg, bus, model, armed=lambda: True)
    detected = bus.subscribe(WakeWordDetected)

    # 6 frames de 512 = 3072 amostras => 2 chunks completos de 1280
    await drive(bus, detector, [frame() for _ in range(6)])

    assert detected.qsize() == 1


@pytest.mark.asyncio
async def test_converte_float32_para_int16_antes_do_predict():
    bus = EventBus()
    model = FakeWakeModel(score=0.0)
    cfg = WakeWordConfig(threshold=0.5, cooldown_s=0)
    detector = WakeWordDetector(cfg, bus, model, armed=lambda: True)

    await drive(bus, detector, [frame(0.5) for _ in range(3)])

    assert len(model.calls) == 1
    chunk = model.calls[0]
    assert chunk.dtype == np.int16
    assert chunk[0] == int(0.5 * 32767)


# --------------------------------------------------------------------------- #
# create_wakeword_model
# --------------------------------------------------------------------------- #


def test_create_wakeword_model_none_quando_desabilitado():
    assert create_wakeword_model(WakeWordConfig(enabled=False)) is None


def test_create_wakeword_model_none_quando_construcao_falha(monkeypatch):
    import elise.audio.wakeword as wakeword_mod

    def boom(cfg):
        raise RuntimeError("modelo indisponível")

    monkeypatch.setattr(wakeword_mod, "OpenWakeWordModel", boom)

    assert create_wakeword_model(WakeWordConfig(enabled=True)) is None


# --------------------------------------------------------------------------- #
# resolve_model_name
# --------------------------------------------------------------------------- #


def test_resolve_model_name_usa_custom_quando_arquivo_existe(tmp_path: Path):
    custom = tmp_path / "hey_elise.onnx"
    custom.write_bytes(b"fake-onnx")
    cfg = WakeWordConfig(model=str(custom), fallback_model="hey_jarvis")
    assert resolve_model_name(cfg) == str(custom)


def test_resolve_model_name_cai_no_fallback_quando_onnx_nao_existe():
    cfg = WakeWordConfig(
        model="models/wakeword/hey_elise.onnx", fallback_model="hey_jarvis"
    )
    assert resolve_model_name(cfg) == "hey_jarvis"


def test_resolve_model_name_nome_pretreinado_nao_verifica_arquivo():
    cfg = WakeWordConfig(model="hey_jarvis", fallback_model="hey_jarvis")
    assert resolve_model_name(cfg) == "hey_jarvis"


# --------------------------------------------------------------------------- #
# OpenWakeWordModel — baixa os .onnx antes de carregar (regressão: Model() não
# baixa sozinho, precisa de openwakeword.utils.download_models() antes)
# --------------------------------------------------------------------------- #


def test_open_wakeword_model_baixa_modelos_antes_de_carregar(monkeypatch):
    ow_model = pytest.importorskip("openwakeword.model")
    ow_utils = pytest.importorskip("openwakeword.utils")

    calls: list[list[str]] = []
    monkeypatch.setattr(
        ow_utils, "download_models", lambda model_names=[]: calls.append(list(model_names))
    )

    class FakeOrtModel:
        def __init__(self, wakeword_models, inference_framework):
            self.models = {wakeword_models[0]: object()}

    monkeypatch.setattr(ow_model, "Model", FakeOrtModel)

    from elise.audio.wakeword import OpenWakeWordModel

    OpenWakeWordModel(WakeWordConfig(model="hey_jarvis"))

    assert calls == [["hey_jarvis"]]
