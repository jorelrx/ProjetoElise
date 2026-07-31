"""Segmentação incremental de sentenças sobre o stream de tokens do LLM.

Técnica-chave de latência (RealtimeTTS/Pipecat): o TTS sintetiza sentença
a sentença enquanto o LLM ainda gera. O tempo até o primeiro áudio passa a
ser ~(primeiro token + primeira sentença + TTS de 1 sentença), tipicamente
bem abaixo de 1 s mesmo em hardware modesto.

Heurísticas pt-BR:
- Quebra em . ! ? … e quebras de linha, exigindo comprimento mínimo para
  não fragmentar em "Oi." / "Sim." isolados quando há continuação.
- Protege abreviações comuns (Sr., Dra., etc.) e números decimais (3.14).
- Remove marcações que o modelo possa emitir (markdown leve) — o texto
  segue para um sintetizador de voz, não para uma tela.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

_ABBREV = {
    "sr", "sra", "srta", "dr", "dra", "prof", "profa", "eng", "av", "r",
    "etc", "ex", "p", "pag", "pág", "tel", "obs", "min", "máx", "max",
}

_SENTENCE_END = re.compile(r"([.!?…]+)(\s+|$)")
_MD_NOISE = re.compile(r"[*_`#>]|(\[(.*?)\]\((?:.*?)\))")


def _clean_for_tts(text: str) -> str:
    text = _MD_NOISE.sub(lambda m: m.group(2) or "", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_boundary(buffer: str, end_idx: int) -> bool:
    """Decide se a pontuação em ``end_idx`` encerra mesmo uma sentença."""
    before = buffer[:end_idx]
    last_word = re.findall(r"[\wÀ-ÿ]+", before)
    if last_word:
        w = last_word[-1].lower()
        if w in _ABBREV:
            return False
        # Número decimal ("3.14") ou ordinal em lista ("1.")
        if w.isdigit() and end_idx < len(buffer) and buffer[end_idx] == ".":
            after = buffer[end_idx + 1 : end_idx + 2]
            if after.isdigit():
                return False
    return True


async def sentences(
    tokens: AsyncIterator[str],
    min_len: int = 24,
    max_len: int = 320,
) -> AsyncIterator[str]:
    """Agrupa deltas de tokens em sentenças completas prontas para o TTS."""
    buffer = ""
    async for tok in tokens:
        buffer += tok
        while True:
            m = _SENTENCE_END.search(buffer)
            if not m:
                # Sentença gigante sem pontuação: corta na última vírgula/espaço.
                if len(buffer) >= max_len:
                    cut = max(buffer.rfind(",", 0, max_len), buffer.rfind(" ", 0, max_len))
                    cut = cut if cut > 0 else max_len
                    chunk = _clean_for_tts(buffer[:cut])
                    buffer = buffer[cut:]
                    if chunk:
                        yield chunk
                    continue
                break
            end = m.end(1)
            if not _is_boundary(buffer, m.start(1)):
                # Falso limite (abreviação/decimal): procura o próximo.
                nxt = _SENTENCE_END.search(buffer, end)
                if nxt is None:
                    break
                m, end = nxt, nxt.end(1)
                if not _is_boundary(buffer, m.start(1)):
                    break
            candidate = buffer[:end]
            if len(candidate.strip()) < min_len and len(buffer) == end:
                break  # curta demais e nada depois ainda: espera mais tokens
            chunk = _clean_for_tts(candidate)
            buffer = buffer[end:]
            if chunk:
                yield chunk
    tail = _clean_for_tts(buffer)
    if tail:
        yield tail
