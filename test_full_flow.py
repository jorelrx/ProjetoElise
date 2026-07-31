#!/usr/bin/env python
"""Teste completo do fluxo LLM + TTS."""

import asyncio
import sys
import io
sys.path.insert(0, 'src')

from elise.config import load_config
from elise.llm import create_llm
from elise.tts import create_tts
from elise.llm.sentence_stream import sentences

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

async def test_full_flow():
    cfg = load_config("config.yaml")
    llm = create_llm(cfg.llm)
    tts = create_tts(cfg.tts)
    
    if not await llm.healthcheck():
        print("Erro: LLM não está acessível")
        return
    
    print("LLM e TTS estão acessíveis. Testando fluxo completo...")
    
    # Teste simples de resposta
    print("\nResposta do LLM:")
    llm._history.clear()
    llm._cfg.system_prompt = "Você é uma assistente."
    async for sentence in sentences(llm.stream_reply("Olá")):
        print(f"Sentença: {sentence}")
        # Sintetizar a sentença
        chunks = [c async for c in tts.synthesize(sentence)]
        print(f"Chunks de áudio: {len(chunks)}")
    
    await llm.aclose()

if __name__ == "__main__":
    asyncio.run(test_full_flow())
