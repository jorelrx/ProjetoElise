"""Configuração tipada e validada da Elise.

Carrega `config.yaml` para modelos Pydantic. Qualquer erro de configuração
falha cedo (fail-fast) com mensagem clara, em vez de estourar em runtime
no meio de uma conversa.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class AudioConfig(BaseModel):
    input_device: int | str | None = None
    sample_rate: int = 16000
    frame_samples: int = 512

    @field_validator("sample_rate")
    @classmethod
    def _must_be_16k(cls, v: int) -> int:
        if v != 16000:
            raise ValueError("sample_rate deve ser 16000 (taxa nativa do Silero VAD e do Whisper)")
        return v

    @property
    def frame_ms(self) -> float:
        return 1000.0 * self.frame_samples / self.sample_rate


class VadConfig(BaseModel):
    threshold: float = Field(0.55, ge=0.0, le=1.0)
    min_speech_ms: int = Field(200, ge=0)
    min_silence_ms: int = Field(700, ge=100)
    pre_roll_ms: int = Field(320, ge=0)
    max_utterance_s: int = Field(30, ge=1)


class DenoiseBackend(str, Enum):
    DEEPFILTERNET = "deepfilternet"
    NOISEREDUCE = "noisereduce"
    NONE = "none"


class DenoiseConfig(BaseModel):
    backend: DenoiseBackend = DenoiseBackend.DEEPFILTERNET


class SttConfig(BaseModel):
    backend: str = "faster_whisper"
    model: str = "small"
    language: str = "pt"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = Field(5, ge=1)


class LlmConfig(BaseModel):
    backend: str = "openai_compat"
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    model: str = "qwen3.5-9b"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=1)
    request_timeout_s: float = Field(120.0, gt=0)
    history_max_turns: int = Field(12, ge=1)
    system_prompt: str = "Você é a Elise, uma assistente de voz brasileira."
    reasoning_effort: str | None = "none"


class PiperConfig(BaseModel):
    model_path: Path = Path("models/piper/pt_BR-faber-medium.onnx")


class EdgeConfig(BaseModel):
    voice: str = "pt-BR-FranciscaNeural"


class TtsConfig(BaseModel):
    backend: str = "piper"
    piper: PiperConfig = PiperConfig()
    edge: EdgeConfig = EdgeConfig()


class DuplexMode(str, Enum):
    HALF = "half_duplex"
    FULL = "full_duplex"


class BehaviorConfig(BaseModel):
    mode: DuplexMode = DuplexMode.HALF


class LoggingConfig(BaseModel):
    level: str = "INFO"


class EliseConfig(BaseModel):
    audio: AudioConfig = AudioConfig()
    vad: VadConfig = VadConfig()
    denoise: DenoiseConfig = DenoiseConfig()
    stt: SttConfig = SttConfig()
    llm: LlmConfig = LlmConfig()
    tts: TtsConfig = TtsConfig()
    behavior: BehaviorConfig = BehaviorConfig()
    logging: LoggingConfig = LoggingConfig()


def load_config(path: str | Path = "config.yaml") -> EliseConfig:
    """Carrega e valida a configuração. Ausência do arquivo => defaults."""
    p = Path(path)
    if not p.exists():
        return EliseConfig()
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return EliseConfig.model_validate(raw)
