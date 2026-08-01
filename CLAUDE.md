# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projeto

Elise — assistente de voz local-first (pt-BR) para Windows 11. Pipeline: microfone → VAD → denoise → STT → LLM (LM Studio) → TTS → alto-falantes. Todo o código, logs, comentários e docs são em português — mantenha esse padrão. O documento de referência das decisões técnicas é `docs/estudo_base.md`.

## Comandos

```powershell
.venv\Scripts\activate           # venv já existe na raiz
pip install -e .[dev]            # dev tools; extras: [denoise] [noisereduce] [edge]

pytest                           # roda só tests/ (testpaths no pyproject)
pytest tests/test_segmenter.py -k nome_do_teste
ruff check src tests
mypy src

elise                            # assistente completa (requer LM Studio servindo em localhost:1234)
elise --text                     # REPL de texto — testa LLM+TTS sem microfone (melhor p/ dev)
elise --list-devices             # índices de dispositivos de áudio
```

Os `test_*.py` na raiz (`test_full_flow.py`, `test_interactive.py`, `test_llm_flow.py`) são scripts manuais de validação, não testes pytest — pytest não os coleta.

Rodar a assistente exige LM Studio com servidor ativo (`http://localhost:1234/v1`, modelo Qwen3.5-9B) e a voz Piper em `models/piper/pt_BR-faber-medium.onnx` (+ `.json`). O healthcheck do LLM roda antes de abrir o microfone e aborta com dica se o servidor estiver fora. O modelo Silero VAD (~2 MB) baixa sozinho para `models/` na primeira execução. `models/` está no `.gitignore`.

## Arquitetura

**Orientada a eventos via `EventBus` asyncio** (`src/elise/events.py`): pub/sub por tipo de evento, cada assinante tem fila própria. Filas de áudio usam `maxsize` com **drop-oldest** — sob pressão, frames antigos são descartados em vez de acumular latência. Eventos principais: `AudioFrame`, `SpeechStarted`, `UtteranceCaptured`.

**Fluxo de um turno** (`src/elise/orchestrator.py`, máquina de estados LISTENING → THINKING → SPEAKING):

1. `audio/capture.py` — callback PortAudio mínimo (só copia buffer e faz handoff thread-safe ao event loop) publica `AudioFrame`.
2. `audio/vad.py` — Silero VAD v5 via onnxruntime + `UtteranceSegmenter` (endpointing com pre-roll); publica `SpeechStarted` e `UtteranceCaptured`. O wrapper `SileroVad` precisa colar as últimas 64 amostras do frame anterior no frame novo (exigência do grafo v5) — sem isso a saída trava perto de 0.
3. `audio/denoise/` — denoise **por enunciado inteiro** (pós-VAD, pré-STT), não em streaming. Cascata de fallback: deepfilternet → noisereduce → passthrough.
4. `stt/` — faster-whisper `small` pt-BR, CPU int8 (CTranslate2 não usa GPU AMD).
5. `llm/` — cliente OpenAI-compatible em streaming SSE (`OpenAICompatChat`). Trocar LM Studio por Ollama/nuvem = trocar `base_url`/`model` no config.
6. `llm/sentence_stream.py` — segmenta o stream de tokens em sentenças (protege abreviações/decimais pt-BR, limpa markdown).
7. `tts/` + `audio/playback.py` — pipelining: produtor sintetiza a sentença N+1 enquanto a N toca (fila `maxsize=2`).

**Composition root**: `__main__.py:_build_services()` instancia todos os backends a partir do `config.yaml` (validado por Pydantic em `config.py`, fail-fast). Cada subsistema tem factory `create_*` que resolve o backend pelo config — novos backends entram por aí, sem tocar no orquestrador.

**Memória conversacional transacional** (`llm/__init__.py`): o turno do usuário entra no histórico ao iniciar o stream, mas a resposta só é gravada via `commit_reply()` com o texto **efetivamente falado**. Barge-in/interrupção → commit truncado com "—"; erro/transcrição vazia → `rollback_user_turn()`. Qualquer mudança no fluxo do turno precisa preservar esse contrato.

**Duplex**: `half_duplex` (padrão) fecha o gate do microfone enquanto a Elise pensa/fala (`Orchestrator.mic_gate_open`, consultado pelo segmenter). `full_duplex` habilita barge-in experimental — sem AEC ainda, a Elise pode se auto-interromper sem fone de ouvido.

## Restrições que não são óbvias

- `sample_rate` deve ser 16000 e `frame_samples` 512 — exigências do Silero VAD v5; o config valida e rejeita outros valores.
- `llm.reasoning_effort: none` no config existe porque Qwen3.x gasta `max_tokens` inteiro em `reasoning_content` e devolve `content` vazio se o thinking não for desligado.
- Trabalho CPU-bound (denoise, STT, TTS) roda em executor — nunca bloquear o event loop, e nunca colocar trabalho pesado no callback de captura de áudio.
- `sys.stdout.reconfigure(encoding="utf-8")` no `main()` é necessário no console do Windows (cp1252 quebra acentos pt-BR).
- KPI de latência: log `turno.primeiro_audio` mede tempo do fim da fala do usuário até o primeiro áudio da resposta.
