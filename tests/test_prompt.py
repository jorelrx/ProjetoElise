from datetime import datetime
from pathlib import Path

import pytest

from elise.config import PromptConfig
from elise.llm.prompt import PromptBuilder


def test_prompt_contem_persona_regras_e_data():
    builder = PromptBuilder(PromptConfig(), clock=lambda: datetime(2024, 1, 1, 8, 0))
    prompt = builder.build()
    assert "Elise" in prompt
    assert "Nunca use markdown" in prompt
    assert "segunda-feira, 1 de janeiro de 2024, 08:00" in prompt


def test_include_datetime_false_omite_contexto_de_data():
    builder = PromptBuilder(
        PromptConfig(include_datetime=False), clock=lambda: datetime(2024, 1, 1, 8, 0)
    )
    prompt = builder.build()
    assert "2024" not in prompt
    assert "segunda-feira" not in prompt


def test_persona_customizada_por_arquivo(tmp_path: Path):
    persona_path = tmp_path / "minha_persona.md"
    persona_path.write_text("PERSONA_TESTE_XYZ", encoding="utf-8")
    builder = PromptBuilder(PromptConfig(persona=str(persona_path)))
    prompt = builder.build()
    assert "PERSONA_TESTE_XYZ" in prompt


def test_persona_inexistente_falha_cedo():
    with pytest.raises(ValueError):
        PromptBuilder(PromptConfig(persona="caminho/que/nao/existe.md"))
