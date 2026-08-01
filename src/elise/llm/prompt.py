"""Construção do system prompt em camadas: persona → regras de voz → contexto.

A saída do LLM vai direto para o TTS — por isso a camada de regras de voz é
fixa e sempre aplicada, independente da persona escolhida. O prompt é
remontado a cada turno (``build()``) para que a camada de contexto dinâmico
(data/hora) fique sempre atual.
"""

from __future__ import annotations

import importlib.resources
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from ..config import PromptConfig

REGRAS_DE_VOZ = """\
Regras de fala (sua resposta vai direto para um sintetizador de voz, não \
para uma tela):
- Nunca use markdown, listas, tabelas, emojis ou qualquer formatação.
- Frases curtas e naturais, como numa conversa falada.
- Seja concisa; evite parágrafos longos.
- Escreva números, siglas e unidades por extenso."""

_DIAS_SEMANA = [
    "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
    "sexta-feira", "sábado", "domingo",
]
_MESES = [
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
]

_PERSONAS_EMBUTIDAS = {"elise"}

def _formatar_data_hora(dt: datetime) -> str:
    dia_semana = _DIAS_SEMANA[dt.weekday()]
    mes = _MESES[dt.month - 1]
    return f"{dia_semana}, {dt.day} de {mes} de {dt.year}, {dt:%H:%M}"


def _carregar_persona(nome_ou_caminho: str) -> str:
    if nome_ou_caminho in _PERSONAS_EMBUTIDAS:
        arquivo = importlib.resources.files("elise.llm") / "personas" / f"{nome_ou_caminho}.md"
        return arquivo.read_text(encoding="utf-8").strip()
    caminho = Path(nome_ou_caminho)
    if not caminho.exists():
        raise ValueError(
            f"Persona '{nome_ou_caminho}' não é embutida ({sorted(_PERSONAS_EMBUTIDAS)}) "
            "nem um arquivo existente."
        )
    return caminho.read_text(encoding="utf-8").strip()

class PromptBuilder:
    """Monta o system prompt: persona → regras de voz → contexto dinâmico."""

    def __init__(
        self,
        cfg: PromptConfig,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._cfg = cfg
        self._clock = clock
        self._persona = _carregar_persona(cfg.persona)

    def build(self) -> str:
        partes = [self._persona, REGRAS_DE_VOZ]
        if self._cfg.include_datetime:
            partes.append(f"Contexto: agora é {_formatar_data_hora(self._clock())}.")
        return "\n\n".join(partes)
