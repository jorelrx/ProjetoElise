# Elise — Assistente de Voz Local-First (Fase 1)

Escuta contínua → redução de ruído → VAD → STT → **Qwen3.5-9B no LM Studio** → TTS,
tudo rodando localmente no Windows 11 (RX 580 8GB friendly), conforme o estudo técnico do projeto.

```
Microfone 16kHz (frames de 32ms, thread de áudio dedicada)
   └─▶ EventBus (asyncio, backpressure drop-oldest)
        └─▶ Silero VAD v5 (ONNX, CPU) → segmentação de enunciados (pre-roll + endpointing)
             └─▶ DeepFilterNet3 (denoise por enunciado; fallback noisereduce → none)
                  └─▶ faster-whisper `small` pt-BR (CPU int8)
                       └─▶ LM Studio /v1/chat/completions (Qwen3.5-9B, streaming SSE)
                            └─▶ segmentação de sentenças em streaming
                                 └─▶ Piper TTS pt_BR (sintetiza a próxima sentença
                                     enquanto a atual toca) → alto-falantes
```

## Requisitos

- Windows 11 (funciona também em Linux/macOS), Python **3.10+**
- **LM Studio** com o servidor local ativo:
  1. Baixe/carregue o modelo **Qwen3.5-9B** (quantização Q4_K_M cabe na RX 580 8GB; use o backend **Vulkan** nas configurações de runtime do LM Studio).
  2. Aba **Developer → Start Server** (padrão `http://localhost:1234/v1`).
  3. Confira o identificador do modelo e ajuste `llm.model` no `config.yaml` se necessário.
- **Voz Piper pt-BR**: baixe `pt_BR-miro-high.onnx` **e** `pt_BR-miro-high.onnx.json` de
  [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/tree/main/pt/pt_BR/miro/high)
  para `models/piper/`.

## Instalação

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .                 # núcleo
pip install -e .[denoise]        # DeepFilterNet3 (recomendado; baixa torch)
# alternativas leves:
pip install -e .[noisereduce]    # denoise leve sem torch
pip install -e .[edge]           # TTS de nuvem (requer ffmpeg no PATH)
pip install -e .[dev]            # pytest / ruff / mypy
```

> O modelo do Silero VAD (~2 MB) é baixado automaticamente na primeira execução para `models/`.

## Uso

```powershell
elise                    # assistente de voz completa
elise --text             # REPL de texto (testa LLM+TTS sem microfone)
elise --list-devices     # descobre o índice do seu microfone
elise --config meu.yaml
```

Tudo é configurável em **`config.yaml`** (backends de denoise/STT/LLM/TTS, limiares do VAD,
modo half/full-duplex, prompt de persona). Trocar LM Studio por Ollama/llama.cpp/nuvem é
trocar `llm.base_url` + `llm.model` — abstração OpenAI-compatible.

## Decisões de engenharia (resumo)

- **Arquitetura orientada a eventos** (`EventBus` asyncio) com serviços desacoplados e
  filas com *drop-oldest* no áudio: sob pressão, descartamos frames antigos em vez de
  acumular latência — comportamento correto para tempo real. Pronta para migrar a
  ZeroMQ/Redis ou a um serviço de áudio nativo (Rust) na Fase 2 sem tocar no restante.
- **Callback de captura mínimo**: na thread de áudio do PortAudio só há cópia do buffer e
  handoff thread-safe ao event loop (regra de ouro de real-time audio).
- **Denoise por enunciado (pós-VAD, pré-STT)**: o Silero é robusto a ruído; quem precisa de
  áudio limpo é o Whisper. Dar o segmento inteiro ao DeepFilterNet3 rende qualidade melhor
  que streaming em blocos de 32 ms e mantém o caminho quente leve. Fallback gracioso em
  cascata (deepfilternet → noisereduce → passthrough) com aviso em log.
- **Latência percebida mínima**: LLM em streaming SSE → segmentador incremental de
  sentenças (com proteção a abreviações/decimais pt-BR e limpeza de markdown) → o TTS
  sintetiza a sentença N+1 enquanto a N toca. O log `turno.primeiro_audio` mede o KPI real.
- **Memória conversacional transacional**: o histórico só registra o que a Elise
  *efetivamente falou* — se você a interromper, o turno é gravado truncado, mantendo o
  contexto fiel ao que você ouviu.
- **Half-duplex por padrão**: sem AEC (Fase 2), o microfone é silenciado enquanto a Elise
  fala, evitando que ela escute a si mesma. `behavior.mode: full_duplex` habilita
  **barge-in experimental** (ideal com fone de ouvido).
- **Fail-fast e observabilidade**: config validada por Pydantic na inicialização,
  healthcheck do LM Studio antes de abrir o microfone, logs estruturados (structlog) com
  latência por estágio.

## Testes e qualidade

```powershell
pytest            # 13 testes: segmentador VAD, sentenças em streaming, config, backpressure
ruff check src tests
mypy src
```

## Roadmap (conforme o estudo)

- **Fase 2 — áudio robusto**: wake word "Hey Elise" (openWakeWord), AEC3 com loopback
  WASAPI (falar por cima de música), DeepFilterNet em streaming, verificação de locutor
  (ECAPA-TDNN), memória vetorial (sqlite-vec).
- **Fase 3 — agêntica**: function calling (Qwen tools) → MCP → controle do PC,
  sempre permission-gated.
