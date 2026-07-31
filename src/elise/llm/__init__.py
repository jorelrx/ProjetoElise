"""Cérebro conversacional — cliente OpenAI-compatible em streaming.

A abstração OpenAI-compatible é a decisão central de arquitetura do
estudo: LM Studio (caso deste projeto, servindo o Qwen3.5-9B em
``http://localhost:1234/v1``), Ollama, llama.cpp ``llama-server`` e as
nuvens (OpenAI/Azure/Groq...) falam o mesmo protocolo. Trocar de backend
é trocar ``base_url``/``model`` no config.yaml.

Streaming SSE é obrigatório para voz: o TTS começa a falar na primeira
sentença completa, muito antes de o modelo terminar a resposta.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import AsyncIterator
from typing import Protocol

import httpx
import structlog

from ..config import LlmConfig

log = structlog.get_logger(__name__)


class ChatModel(Protocol):
    async def stream_reply(self, user_text: str) -> AsyncIterator[str]: ...

    def commit_reply(self, text: str) -> None: ...

    def rollback_user_turn(self) -> None: ...


class OpenAICompatChat:
    """Cliente /v1/chat/completions com memória de curto prazo em janela.

    A memória é *transacional*: o turno do usuário entra no histórico ao
    iniciar o streaming, mas a resposta só é gravada via ``commit_reply``
    quando falada até o fim. Se o usuário interromper (barge-in), o
    orquestrador chama ``commit_reply`` com o trecho efetivamente falado —
    o histórico reflete o que o usuário de fato ouviu.
    """

    def __init__(self, cfg: LlmConfig) -> None:
        self._cfg = cfg
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url,
            timeout=httpx.Timeout(cfg.request_timeout_s, connect=10.0),
            headers={"Authorization": f"Bearer {cfg.api_key}"},
        )
        self._history: deque[dict[str, str]] = deque(maxlen=cfg.history_max_turns * 2)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def healthcheck(self) -> bool:
        """Verifica se o servidor (LM Studio) está de pé e com modelo carregado."""
        try:
            r = await self._client.get("/models")
            r.raise_for_status()
            models = [m.get("id") for m in r.json().get("data", [])]
            log.info("llm.servidor_ok", base_url=self._cfg.base_url, modelos=models)
            if self._cfg.model not in models:
                log.warning(
                    "llm.modelo_nao_listado",
                    esperado=self._cfg.model,
                    dica="confira o identificador do modelo carregado no LM Studio",
                )
            return True
        except Exception as exc:  # noqa: BLE001
            log.error(
                "llm.servidor_inacessivel",
                base_url=self._cfg.base_url,
                erro=str(exc),
                dica="abra o LM Studio > Developer > Start Server (porta 1234)",
            )
            return False

    # ------------------------------------------------------------------ #

    async def stream_reply(self, user_text: str) -> AsyncIterator[str]:
        self._history.append({"role": "user", "content": user_text})
        payload = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": self._cfg.system_prompt},
                *self._history,
            ],
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
            "stream": True,
        }
        if self._cfg.reasoning_effort:
            # Modelos híbridos (Qwen3.x) gastam max_tokens inteiro em
            # "reasoning_content" antes de emitir a resposta em "content" —
            # sem isto, a resposta pode sair vazia. LM Studio/llama.cpp aceita
            # este campo para desligar o raciocínio em cadeia.
            payload["reasoning_effort"] = self._cfg.reasoning_effort
        got_content = False
        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                token = delta.get("content")
                if token:
                    got_content = True
                    yield token
        if not got_content:
            log.warning(
                "llm.resposta_vazia",
                dica="modelo gastou o max_tokens em reasoning_content sem emitir content; "
                "aumente max_tokens ou confira reasoning_effort no config.yaml",
            )

    def commit_reply(self, text: str) -> None:
        text = text.strip()
        if text:
            self._history.append({"role": "assistant", "content": text})

    def rollback_user_turn(self) -> None:
        if self._history and self._history[-1]["role"] == "user":
            self._history.pop()


def create_llm(cfg: LlmConfig) -> OpenAICompatChat:
    if cfg.backend == "openai_compat":
        return OpenAICompatChat(cfg)
    raise ValueError(f"Backend de LLM desconhecido: {cfg.backend}")
