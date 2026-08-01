import json

import httpx
import pytest

from elise.config import LlmConfig
from elise.llm import OpenAICompatChat


def _sse_handler(request: httpx.Request) -> httpx.Response:
    body = (
        "data: " + json.dumps({"choices": [{"delta": {"content": "Oi"}}]}) + "\n\n"
        "data: [DONE]\n\n"
    )
    return httpx.Response(200, text=body)


@pytest.mark.asyncio
async def test_stream_reply_usa_system_prompt_provider():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _sse_handler(request)

    chat = OpenAICompatChat(LlmConfig(), system_prompt_provider=lambda: "PROMPT_DINAMICO")
    chat._client = httpx.AsyncClient(
        base_url=chat._cfg.base_url, transport=httpx.MockTransport(handler)
    )

    tokens = [t async for t in chat.stream_reply("oi")]

    assert tokens == ["Oi"]
    assert captured["payload"]["messages"][0] == {
        "role": "system",
        "content": "PROMPT_DINAMICO",
    }
