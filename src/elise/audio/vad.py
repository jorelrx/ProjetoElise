"""VAD (Silero v5 via ONNX Runtime) + segmentador de enunciados.

Silero VAD é a recomendação do estudo técnico: deep-learning, robusto a
ruído, roda em CPU com custo desprezível. Usamos o modelo ONNX direto no
onnxruntime (sem PyTorch) — download automático na primeira execução.

O segmentador implementa a máquina de estados clássica de endpointing:

    SILÊNCIO --(prob>=thr por min_speech_ms)--> FALA   [emite SpeechStarted]
    FALA     --(prob<thr por min_silence_ms)--> SILÊNCIO [emite UtteranceCaptured]

com pre-roll (não corta o ataque da primeira palavra) e teto de duração.
"""

from __future__ import annotations

import asyncio
import time
import urllib.request
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
import structlog

from ..config import AudioConfig, VadConfig
from ..events import AudioFrame, EventBus, SpeechStarted, UtteranceCaptured

log = structlog.get_logger(__name__)

SILERO_ONNX_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    "src/silero_vad/data/silero_vad.onnx"
)


def ensure_silero_model(models_dir: Path = Path("models")) -> Path:
    """Garante o silero_vad.onnx em disco (download único, ~2 MB)."""
    path = models_dir / "silero_vad.onnx"
    if not path.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        log.info("vad.baixando_modelo", url=SILERO_ONNX_URL)
        urllib.request.urlretrieve(SILERO_ONNX_URL, path)  # noqa: S310
    return path


class SileroVad:
    """Wrapper stateful do Silero VAD v5 (frames de 512 amostras @16 kHz).

    O grafo combinado v5 espera, a cada chamada, as últimas ``CONTEXT``
    amostras do frame anterior coladas na frente do frame novo (ver
    ``OnnxWrapper`` do pacote oficial ``silero-vad``). Sem isso o modelo
    nunca vê a continuidade do sinal e a saída fica presa perto de 0
    independente do conteúdo do áudio.
    """

    FRAME = 512
    CONTEXT = 64

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(16000, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Retorna a probabilidade de fala (0-1) de um frame de 512 amostras."""
        if frame.shape[0] != self.FRAME:
            raise ValueError(f"Silero VAD exige frames de {self.FRAME} amostras")
        x = np.concatenate([self._context, frame.reshape(1, -1).astype(np.float32)], axis=1)
        out, self._state = self._session.run(
            None,
            {
                "input": x,
                "state": self._state,
                "sr": self._sr,
            },
        )
        self._context = x[:, -self.CONTEXT :]
        return float(out[0][0])


class UtteranceSegmenter:
    """Consome AudioFrame do bus e emite SpeechStarted / UtteranceCaptured.

    ``vad`` é injetável (Callable[frame]->prob), o que permite testar a
    máquina de estados sem ONNX e trocar o VAD (WebRTC, TEN) no futuro.

    ``gate`` controla half-duplex: quando fechado (Elise falando), os
    frames são ignorados e o estado interno é zerado — o microfone "não
    existe" durante a fala da assistente.
    """

    def __init__(
        self,
        audio_cfg: AudioConfig,
        vad_cfg: VadConfig,
        bus: EventBus,
        vad: Callable[[np.ndarray], float],
        gate_open: Callable[[], bool] = lambda: True,
    ) -> None:
        self._cfg = vad_cfg
        self._bus = bus
        self._vad = vad
        self._gate_open = gate_open

        frame_ms = audio_cfg.frame_ms
        self._frames_speech = max(1, int(vad_cfg.min_speech_ms / frame_ms))
        self._frames_silence = max(1, int(vad_cfg.min_silence_ms / frame_ms))
        self._frames_preroll = max(1, int(vad_cfg.pre_roll_ms / frame_ms))
        self._frames_max = int(vad_cfg.max_utterance_s * 1000 / frame_ms)
        self._sample_rate = audio_cfg.sample_rate

        self._queue = bus.subscribe(AudioFrame, maxsize=64)
        self._reset_utterance()

    # ------------------------------------------------------------------ #

    def _reset_utterance(self) -> None:
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._preroll: deque[np.ndarray] = deque(maxlen=self._frames_preroll)
        self._buffer: list[np.ndarray] = []

    async def run(self) -> None:
        while True:
            frame: AudioFrame = await self._queue.get()
            if not self._gate_open():
                if self._in_speech or self._preroll:
                    self._reset_utterance()
                    if hasattr(self._vad, "reset"):
                        self._vad.reset()  # type: ignore[attr-defined]
                continue
            self._process(frame)

    # ------------------------------------------------------------------ #

    def _process(self, frame: AudioFrame) -> None:
        prob = self._vad(frame.samples)
        is_speech = prob >= self._cfg.threshold

        if not self._in_speech:
            self._preroll.append(frame.samples)
            self._speech_run = self._speech_run + 1 if is_speech else 0
            if self._speech_run >= self._frames_speech:
                self._in_speech = True
                self._silence_run = 0
                self._buffer = list(self._preroll)
                self._bus.publish(SpeechStarted(timestamp=time.monotonic()))
                log.debug("vad.fala_iniciada", prob=round(prob, 2))
            return

        # Em fala: acumula tudo (inclusive silêncios curtos entre palavras).
        self._buffer.append(frame.samples)
        self._silence_run = 0 if is_speech else self._silence_run + 1

        ended = self._silence_run >= self._frames_silence
        too_long = len(self._buffer) >= self._frames_max
        if ended or too_long:
            audio = np.concatenate(self._buffer)
            # Remove a cauda de silêncio final (mantém ~120 ms de respiro).
            if ended:
                keep_tail = int(0.12 * self._sample_rate)
                cut = max(0, self._silence_run * SileroVad.FRAME - keep_tail)
                if cut:
                    audio = audio[:-cut]
            duration = audio.shape[0] / self._sample_rate
            self._reset_utterance()
            if hasattr(self._vad, "reset"):
                self._vad.reset()  # type: ignore[attr-defined]
            if duration < self._cfg.min_speech_ms / 1000.0:
                return
            log.info("vad.enunciado_capturado", duracao_s=round(duration, 2))
            self._bus.publish(UtteranceCaptured(audio=audio, duration_s=duration))


async def run_segmenter_forever(segmenter: UtteranceSegmenter) -> None:
    try:
        await segmenter.run()
    except asyncio.CancelledError:  # shutdown limpo
        raise
