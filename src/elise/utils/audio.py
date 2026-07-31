"""Conversões de formato de áudio (sem dependências pesadas)."""

from __future__ import annotations

import numpy as np


def float32_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def int16_to_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 32768.0


def rms_dbfs(x: np.ndarray) -> float:
    """Nível RMS em dBFS de um bloco float32 (-inf..0)."""
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    if rms <= 1e-9:
        return -120.0
    return 20.0 * float(np.log10(rms))
