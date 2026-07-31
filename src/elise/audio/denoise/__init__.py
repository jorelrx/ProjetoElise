"""Redução de ruído (speech enhancement) — backends plugáveis.

Decisão de engenharia (documentada no estudo técnico):
o denoise é aplicado **por enunciado**, entre o VAD e o STT, e não frame a
frame no caminho quente do microfone. Motivos:

1. O Silero VAD já é robusto a ruído — não precisa de áudio limpo.
2. O que precisa de áudio limpo é o Whisper, que consome o enunciado
   inteiro de uma vez; limpar o segmento completo dá ao DeepFilterNet
   contexto máximo e melhor qualidade do que streaming em blocos de 32 ms.
3. Mantém a thread/loop de captura leve e o orçamento de latência
   previsível (o denoise de 2-5 s de fala leva dezenas de ms em CPU).

O caminho streaming (AEC3 + DeepFilterNet frame a frame, com loopback
WASAPI como referência) é a Fase 2 do roadmap, no serviço de áudio nativo.

Backends (fallback em cascata se a dependência não estiver instalada):
- ``deepfilternet``: DeepFilterNet3 — SOTA open-source em ruído complexo
  (música/fala de fundo), substituto do RTX Voice em GPUs AMD.
- ``noisereduce``: spectral gating leve, sem torch.
- ``none``: passthrough.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import structlog

from ...config import DenoiseBackend, DenoiseConfig

log = structlog.get_logger(__name__)


class Denoiser(Protocol):
    name: str

    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Recebe/retorna float32 mono no mesmo sample_rate."""
        ...


class PassthroughDenoiser:
    name = "none"

    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return audio


class NoiseReduceDenoiser:
    """Spectral gating (noisereduce) — leve, sem GPU/torch."""

    name = "noisereduce"

    def __init__(self) -> None:
        import noisereduce  # falha aqui se não instalado

        self._nr = noisereduce

    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        out = self._nr.reduce_noise(
            y=audio, sr=sample_rate, stationary=False, prop_decrease=0.85
        )
        return out.astype(np.float32)


class DeepFilterNetDenoiser:
    """DeepFilterNet3 — qualidade superior em ruído não-estacionário."""

    name = "deepfilternet"

    def __init__(self) -> None:
        from df.enhance import enhance, init_df

        self._model, self._df_state, _ = init_df(log_level="WARNING")
        self._enhance = enhance
        self._df_sr: int = self._df_state.sr()  # 48000

    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch

        # DeepFilterNet opera a 48 kHz: upsample -> enhance -> downsample.
        x = torch.from_numpy(audio).unsqueeze(0)
        if sample_rate != self._df_sr:
            import torchaudio

            x = torchaudio.functional.resample(x, sample_rate, self._df_sr)
        y = self._enhance(self._model, self._df_state, x)
        if sample_rate != self._df_sr:
            import torchaudio

            y = torchaudio.functional.resample(y, self._df_sr, sample_rate)
        return y.squeeze(0).numpy().astype(np.float32)


_CASCADE: dict[DenoiseBackend, list[type]] = {
    DenoiseBackend.DEEPFILTERNET: [DeepFilterNetDenoiser, NoiseReduceDenoiser, PassthroughDenoiser],
    DenoiseBackend.NOISEREDUCE: [NoiseReduceDenoiser, PassthroughDenoiser],
    DenoiseBackend.NONE: [PassthroughDenoiser],
}


def create_denoiser(cfg: DenoiseConfig) -> Denoiser:
    """Instancia o backend pedido, com fallback gracioso em cascata."""
    for cls in _CASCADE[cfg.backend]:
        try:
            d = cls()
            if d.name != cfg.backend.value:
                log.warning(
                    "denoise.fallback",
                    pedido=cfg.backend.value,
                    usando=d.name,
                    dica="instale o extra correspondente: pip install elise[denoise]",
                )
            else:
                log.info("denoise.backend", usando=d.name)
            return d
        except Exception as exc:  # noqa: BLE001 - fallback intencional
            log.debug("denoise.indisponivel", backend=cls.__name__, erro=str(exc))
    return PassthroughDenoiser()  # inalcançável, por segurança
