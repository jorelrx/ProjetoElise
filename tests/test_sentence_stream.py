import pytest

from elise.llm.sentence_stream import sentences


async def _stream(parts):
    for p in parts:
        yield p


async def collect(parts, **kw):
    return [s async for s in sentences(_stream(parts), **kw)]


@pytest.mark.asyncio
async def test_quebra_em_sentencas_com_tokens_fragmentados():
    tokens = ["Oi, tudo be", "m? Eu estou", " ótima hoje.", " E você, como vai?"]
    out = await collect(tokens)
    assert out == ["Oi, tudo bem?", "Eu estou ótima hoje.", "E você, como vai?"]


@pytest.mark.asyncio
async def test_nao_quebra_em_abreviacao_e_decimal():
    tokens = ["O Dr. Silva mediu 3.14 metros de corda, acredita nisso?"]
    out = await collect(tokens)
    assert out == ["O Dr. Silva mediu 3.14 metros de corda, acredita nisso?"]


@pytest.mark.asyncio
async def test_remove_markdown_para_fala():
    tokens = ["Isso é **muito** importante, `sabia`?"]
    out = await collect(tokens)
    assert out == ["Isso é muito importante, sabia?"]


@pytest.mark.asyncio
async def test_flush_do_resto_sem_pontuacao():
    out = await collect(["então é isso aí"])
    assert out == ["então é isso aí"]


@pytest.mark.asyncio
async def test_corta_sentenca_gigante_sem_pontuacao():
    longo = "palavra " * 100
    out = await collect([longo], max_len=120)
    assert len(out) >= 2
    assert all(len(s) <= 130 for s in out)
