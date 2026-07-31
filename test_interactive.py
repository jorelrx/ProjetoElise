#!/usr/bin/env python
"""Teste interativo simples."""

import asyncio
import sys
import io
sys.path.insert(0, 'src')

from elise.config import load_config
from elise.llm import create_llm
from elise.tts import create_tts
from elise.llm.sentence_stream import sentences

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

async def test_interactive():
    cfg = load_config("config.yaml")
    llm = create_llm(cfg.llm)
    tts = create_tts(cfg.tts)
    
    if not await llm.healthcheck():
        print("Erro: LLM não está acessível")
        return
    
    print("Teste interativo. Digite 'sair' para encerrar.")
    
    while True:
        user = input("voce> ").strip()
        if user.lower() in {"sair", "exit", "quit"}:
            break
        if not user:
            continue
        
        print("elise> ", end="", flush=True)
        llm._history.clear()
        llm._cfg.system_prompt = "Você é uma assistente."
        async for sentence in sentences(llm.stream_reply(user)):
            print(sentence, end=" ", flush=True)
        print()
        # Debug: imprimir histórico
        print(f"Histórico: {list(llm._history)}")
    
    await llm.aclose()

if __name__ == "__main__":
    asyncio.run(test_interactive())
