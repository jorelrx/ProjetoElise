"""Ponto de entrada da Elise.

Modos:
  elise                     # assistente de voz completa (mic → voz)
  elise --text              # REPL de texto (testa LLM+TTS sem microfone)
  elise --list-devices      # lista dispositivos de áudio
  elise --config outro.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys

import structlog

from .config import EliseConfig, load_config
from .events import EventBus
from .logsetup import get_logger, setup_logging

log = get_logger(__name__)

BANNER = r"""
  ---------------------------------------------
    E L I S E  -  assistente de voz local-first
    fale naturalmente - Ctrl+C para sair
  ---------------------------------------------
"""


def _build_services(cfg: EliseConfig, bus: EventBus):
    """Composition root: instancia todos os backends a partir do config."""
    from .audio.denoise import create_denoiser
    from .audio.playback import AudioPlayer
    from .llm import create_llm
    from .stt import create_stt
    from .tts import create_tts

    denoiser = create_denoiser(cfg.denoise)
    stt = create_stt(cfg.stt)
    llm = create_llm(cfg.llm)
    tts = create_tts(cfg.tts)
    player = AudioPlayer()
    from .orchestrator import Orchestrator

    orch = Orchestrator(cfg, bus, denoiser, stt, llm, tts, player)
    return orch, llm


async def run_voice(cfg: EliseConfig) -> None:
    from .audio.capture import MicrophoneCapture
    from .audio.vad import SileroVad, UtteranceSegmenter, ensure_silero_model

    bus = EventBus()
    orch, llm = _build_services(cfg, bus)

    if not await llm.healthcheck():
        log.error("abortando", motivo="LLM inacessível — inicie o servidor do LM Studio")
        return

    vad = SileroVad(ensure_silero_model())
    segmenter = UtteranceSegmenter(
        cfg.audio, cfg.vad, bus, vad, gate_open=orch.mic_gate_open
    )

    loop = asyncio.get_running_loop()
    mic = MicrophoneCapture(cfg.audio, bus, loop)
    mic.start()
    print(BANNER)
    log.info("elise.pronta", modo=cfg.behavior.mode.value, llm=cfg.llm.model)

    tasks = [
        asyncio.create_task(segmenter.run(), name="segmenter"),
        asyncio.create_task(orch.run(), name="orchestrator"),
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        mic.stop()
        for t in tasks:
            t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*tasks, return_exceptions=True)
        await llm.aclose()


async def run_text_repl(cfg: EliseConfig) -> None:
    """Modo texto: valida LLM + TTS sem depender de microfone/VAD/STT."""
    from .audio.playback import AudioPlayer
    from .llm import create_llm
    from .llm.sentence_stream import sentences
    from .tts import create_tts

    llm = create_llm(cfg.llm)
    if not await llm.healthcheck():
        return
    tts = create_tts(cfg.tts)
    player = AudioPlayer()
    print(BANNER)
    print("(modo texto — digite e Enter; 'sair' encerra)\n")
    try:
        while True:
            try:
                user = input("voce> ").strip()
            except EOFError:
                break
            if not user:
                continue
            if user.lower() in {"sair", "exit", "quit"}:
                break
            spoken: list[str] = []
            async for sentence in sentences(llm.stream_reply(user)):
                print(f"elise> {sentence}")
                chunks = [c async for c in tts.synthesize(sentence)]

                async def _ait(cs=chunks):
                    for c in cs:
                        yield c

                await player.play(_ait(), tts.sample_rate)
                spoken.append(sentence)
            llm.commit_reply(" ".join(spoken))
    finally:
        await llm.aclose()


def main() -> None:
    with contextlib.suppress(AttributeError):
        # Console do Windows por padrão não é UTF-8 — sem isto, acentos
        # saem trocados ou o print quebra (cp1252/cp437 não cobre pt-BR).
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="elise", description="Assistente de voz local-first")
    parser.add_argument("--config", default="config.yaml", help="caminho do config.yaml")
    parser.add_argument("--text", action="store_true", help="REPL de texto (sem microfone)")
    parser.add_argument("--list-devices", action="store_true", help="lista dispositivos de áudio")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)
    structlog.contextvars.bind_contextvars(app="elise")

    if args.list_devices:
        from .audio.capture import MicrophoneCapture

        print(MicrophoneCapture.list_devices())
        return

    runner = run_text_repl(cfg) if args.text else run_voice(cfg)
    try:
        asyncio.run(runner)
    except KeyboardInterrupt:
        print("\naté mais! 👋")
        sys.exit(0)


if __name__ == "__main__":
    main()
