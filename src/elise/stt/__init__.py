"""Speech-to-Text — interface plugável + backends faster-whisper e whisper.cpp.

faster-whisper (CTranslate2) é a recomendação padrão do estudo: até 4x
mais rápido que o openai/whisper com a mesma acurácia. Na RX 580 o
CTranslate2 não acelera por Vulkan/DirectML, então rodamos CPU int8 com o
modelo ``small`` (pt-BR é idioma Tier-1 do Whisper; WER competitivo).

Para GPU na RX 580 (Polaris/GCN4, sem ROCm), o backend ``whisper_cpp`` usa
o whisper.cpp compilado com Vulkan via bindings ``pywhispercpp`` — ver
extra ``[whispercpp]`` no pyproject e ``docs/estudo_base.md``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from collections.abc import Callable, Iterator
from typing import Protocol

import numpy as np
import structlog

from ..config import SttConfig

log = structlog.get_logger(__name__)


@contextlib.contextmanager
def _capturar_stderr_nativo() -> Iterator[Callable[[], str]]:
    """Captura stderr no nível de descritor de arquivo (fd 2) durante o bloco.

    whisper.cpp escreve seus diagnósticos de carga (inclusive se o Vulkan foi
    encontrado) via fprintf(stderr, ...) na camada C — isso não passa por
    sys.stderr, então só dup2 no fd intercepta. A saída original é devolvida
    ao console ao sair do bloco.
    """
    fd = 2
    salvo = os.dup(fd)
    r, w = os.pipe()
    os.dup2(w, fd)
    os.close(w)
    capturado = bytearray()

    def ler() -> str:
        os.set_blocking(r, False)
        with contextlib.suppress(BlockingIOError):
            while chunk := os.read(r, 65536):
                capturado.extend(chunk)
        return capturado.decode(errors="replace")

    try:
        yield ler
    finally:
        os.dup2(salvo, fd)
        os.close(salvo)
        os.close(r)
        if capturado:
            sys.stderr.write(capturado.decode(errors="replace"))


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
        # self.model.device é o device *efetivamente* resolvido pelo CTranslate2
        # (relevante quando cfg.device="auto") — confirma CPU/GPU sem adivinhar.
        log.info(
            "stt.carregado",
            backend="faster_whisper",
            device=self._model.model.device,
            device_index=self._model.model.device_index,
            compute_type=self._model.model.compute_type,
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


class WhisperCppStt:
    def __init__(self, cfg: SttConfig) -> None:
        from pywhispercpp.model import Model

        if cfg.model_path is None:
            raise ValueError(
                "stt.model_path é obrigatório para o backend whisper_cpp "
                "(caminho do modelo GGML, ex. models/whisper_cpp/ggml-small.bin)"
            )
        log.info("stt.carregando", modelo=str(cfg.model_path), backend="whisper_cpp")
        self._cfg = cfg
        # use_gpu=True cai silenciosamente para CPU se a lib Vulkan não estiver
        # disponível — captura o log nativo de carga (fd stderr) pra confirmar
        # qual backend (Vulkan ou CPU) o whisper.cpp realmente escolheu.
        with _capturar_stderr_nativo() as ler_stderr:
            self._model = Model(
                str(cfg.model_path),
                language=cfg.language,
                params_sampling_strategy=1,  # beam search (0 = greedy)
                beam_search={"beam_size": cfg.beam_size, "patience": -1.0},
                no_context=True,  # cada enunciado é transcrito isolado (sem contexto entre turnos)
                print_progress=False,
                print_realtime=False,
                context_params={"use_gpu": True},  # RX 580 via Vulkan; cai p/ CPU sem lib Vulkan
            )
            saida_carga = ler_stderr()
        gpu_ativa = "whisper_backend_init_gpu: using" in saida_carga
        dispositivo = next(
            (
                linha.split("=", 1)[1].split("|", 1)[0].strip()
                for linha in saida_carga.splitlines()
                if "ggml_vulkan: 0 =" in linha
            ),
            "cpu",
        )
        log.info(
            "stt.carregado",
            backend="whisper_cpp",
            gpu_ativa=gpu_ativa,
            dispositivo=dispositivo if gpu_ativa else "cpu",
        )
        if not gpu_ativa:
            log.warning(
                "stt.whisper_cpp.sem_gpu",
                motivo="Vulkan não encontrado ou lib não compilada com GGML_VULKAN — "
                "rodando em CPU. Ver scripts/fix_pywhispercpp_vulkan.sh.",
            )
        # Serializa transcrições: o modelo não é reentrante e a GPU é uma só.
        self._sem = asyncio.Semaphore(1)

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        assert sample_rate == 16000, "whisper.cpp espera 16 kHz"
        async with self._sem:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        segments = self._model.transcribe(audio.astype(np.float32))
        return " ".join(s.text.strip() for s in segments).strip()


def create_stt(cfg: SttConfig) -> SpeechToText:
    if cfg.backend == "faster_whisper":
        return FasterWhisperStt(cfg)
    if cfg.backend == "whisper_cpp":
        return WhisperCppStt(cfg)
    raise ValueError(f"Backend de STT desconhecido: {cfg.backend}")
