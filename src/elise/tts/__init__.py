"""Text-to-Speech — interface plugável + backends Piper (local) e edge-tts.

Piper (MIT, pt_BR-faber-medium) é o padrão do estudo: VITS ~15M params em
ONNX, tempo real folgado em CPU, ~40 ms até o primeiro áudio. edge-tts é a
opção de nuvem gratuita com vozes Neural pt-BR de altíssima qualidade
(requer ffmpeg no PATH para decodificar o MP3 recebido).

Contrato: ``synthesize(text)`` produz chunks PCM int16 mono no
``sample_rate`` do backend, de forma assíncrona e cancelável.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

import numpy as np
import structlog

from ..config import TtsConfig

log = structlog.get_logger(__name__)


class TextToSpeech(Protocol):
    sample_rate: int

    def synthesize(self, text: str) -> AsyncIterator[np.ndarray]: ...


class PiperTts:
    def __init__(self, cfg: TtsConfig) -> None:
        from piper import PiperVoice

        path = cfg.piper.model_path
        if not path.exists():
            raise FileNotFoundError(
                f"Voz Piper não encontrada em '{path}'. Baixe o .onnx e o .onnx.json de "
                "https://huggingface.co/rhasspy/piper-voices (pt/pt_BR/faber/medium) "
                "e ajuste tts.piper.model_path no config.yaml."
            )
        self._voice = PiperVoice.load(str(path))
        self.sample_rate: int = int(self._voice.config.sample_rate)
        log.info("tts.piper_carregado", voz=path.name, sample_rate=self.sample_rate)

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=8)

        def worker() -> None:
            try:
                if hasattr(self._voice, "synthesize_stream_raw"):  # piper-tts <= 1.2
                    for raw in self._voice.synthesize_stream_raw(text):
                        chunk = np.frombuffer(raw, dtype=np.int16)
                        asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
                else:  # piper-tts >= 1.3 (AudioChunk API)
                    for ac in self._voice.synthesize(text):
                        chunk = np.frombuffer(ac.audio_int16_bytes, dtype=np.int16)
                        asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        task = loop.run_in_executor(None, worker)
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            await task


class EdgeTts:
    """Vozes Neural da Microsoft via edge-tts (nuvem, gratuito).

    O serviço devolve MP3 24 kHz; decodificamos com ffmpeg para PCM int16.
    """

    sample_rate = 24000

    def __init__(self, cfg: TtsConfig) -> None:
        import shutil

        import edge_tts  # noqa: F401 - valida instalação

        if shutil.which("ffmpeg") is None:
            raise RuntimeError("edge-tts requer ffmpeg no PATH para decodificar áudio")
        self._voice = cfg.edge.voice

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        import edge_tts

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", str(self.sample_rate), "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        assert proc.stdin and proc.stdout

        async def feed() -> None:
            try:
                communicate = edge_tts.Communicate(text, self._voice)
                async for msg in communicate.stream():
                    if msg["type"] == "audio":
                        proc.stdin.write(msg["data"])
                        await proc.stdin.drain()
            finally:
                proc.stdin.close()

        feeder = asyncio.ensure_future(feed())
        try:
            while True:
                data = await proc.stdout.read(4096)
                if not data:
                    break
                yield np.frombuffer(data, dtype=np.int16)
        finally:
            feeder.cancel()
            if proc.returncode is None:
                proc.kill()
            await proc.wait()


def create_tts(cfg: TtsConfig) -> TextToSpeech:
    if cfg.backend == "piper":
        return PiperTts(cfg)
    if cfg.backend == "edge":
        return EdgeTts(cfg)
    raise ValueError(f"Backend de TTS desconhecido: {cfg.backend}")
