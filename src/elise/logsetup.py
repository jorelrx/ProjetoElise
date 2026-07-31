"""Logging estruturado (structlog) com saída legível no console.

Logs estruturados (chave=valor) tornam trivial medir latência de cada
estágio do pipeline (STT, primeiro token do LLM, primeiro áudio do TTS),
que é a métrica que governa a experiência de um assistente de voz.
"""

from __future__ import annotations

import logging

import structlog


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level.upper(), format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S.%f"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level.upper())
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
