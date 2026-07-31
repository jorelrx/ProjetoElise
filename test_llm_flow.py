#!/usr/bin/env python
"""Teste simples do fluxo LLM para verificar se o modelo está funcionando."""

import asyncio
import sys
import io
sys.path.insert(0, 'src')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from elise.config import load_config
from elise.llm import create_llm

async def test_llm():
    cfg = load_config("config.yaml")
    llm = create_llm(cfg.llm)
    
    if not await llm.healthcheck():
        print("Erro: LLM não está acessível")
        return
    
    print("LLM está acessível. Testando resposta...")
    
    # Teste simples de resposta
    print("\nResposta do LLM:")
    llm._history.clear()
    async for chunk in llm.stream_reply("Olá"):
        print(chunk, end="", flush=True)
    print("\n\nFim da resposta")
    
    # Teste com prompt simples
    print("\n\nTeste com prompt simples:")
    llm._history.clear()
    llm._cfg.system_prompt = "Você é uma assistente."
    async for chunk in llm.stream_reply("Olá"):
        print(chunk, end="", flush=True)
    print("\n\nFim da resposta")
    
    await llm.aclose()

if __name__ == "__main__":
    asyncio.run(test_llm())
