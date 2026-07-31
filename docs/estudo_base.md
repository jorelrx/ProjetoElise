# Elise: Estudo Técnico para um Assistente de Voz Local-First "Jarvis-style" no Windows 11 (AMD RX 580 8GB)

## TL;DR
- **É viável construir a Elise 100% local numa RX 580 8GB**, mas com escolhas de arquitetura precisas: use o backend **Vulkan** do llama.cpp (não ROCm, que a AMD abandonou no Polaris/GCN4) para o LLM (~15-16 tokens/s em modelos 7B/8B Q4), **faster-whisper via DirectML ou whisper.cpp Vulkan** para STT, **Piper** (pt-BR, CPU) ou **Kokoro** para TTS, e **openWakeWord** para o "Hey Elise". Espere latência realista de ~1,5-2,5s ponta-a-ponta, não os <1s de setups NVIDIA.
- **O requisito crítico — falar por cima de música tocando no mesmo PC — resolve-se com Cancelamento de Eco Acústico (AEC) usando o loopback WASAPI como sinal de referência**, não com separação de fontes (Demucs/Spleeter são inviáveis em tempo real com baixa latência). A melhor combinação é AEC (WebRTC AEC3 ou o AEC nativo do Windows 11 via `IAcousticEchoCancellationControl`) → supressão de ruído (DeepFilterNet) → VAD (Silero) → verificação de locutor (ECAPA-TDNN) para confirmar que é você.
- **Adote uma arquitetura modular orientada a eventos com abstração OpenAI-compatible** para trocar Ollama/LM Studio/nuvem por configuração. Estude o **RealtimeVoiceChat (KoljaB)** como referência mais próxima do objetivo e o **Pipecat** como framework de orquestração. Faça em 3 fases: (1) MVP conversacional, (2) memória + persona, (3) ferramentas + controle do PC com permissões.

## Key Findings

1. **AMD/Vulkan é o caminho, não CUDA nem ROCm.** A RX 580 (Polaris/GCN4) perdeu suporte oficial ROCm da AMD; o Ollama não suporta Vulkan nativamente, então o motor de referência é o **llama.cpp compilado com `-DGGML_VULKAN=ON`** (via `llama-server`). Benchmarks reais: ~15-16 tok/s para Mistral 7B / Llama 3 8B Q4_K_M. Whisper.cpp também acelera por Vulkan, e faster-whisper roda por DirectML/ONNX Runtime na AMD.

2. **"Falar por cima da música" é um problema de AEC, não de separação de fontes.** O loopback WASAPI fornece exatamente o sinal de referência (far-end) necessário. Demucs/Spleeter têm latência alta demais (dezenas a centenas de ms, mais processamento pesado) e são para uso offline.

3. **Português brasileiro tem bom suporte em todos os componentes.** Whisper transcreve pt-BR com WER baixo; Qwen (2.5/3) é o melhor LLM local para pt-BR em 8GB; Piper tem vozes pt-BR nativas (faber-medium, edresson) e Kokoro/XTTS oferecem qualidade superior.

4. **Frameworks maduros existem para inspirar/fundamentar.** RealtimeSTT/RealtimeTTS + RealtimeVoiceChat (MIT) são a base local mais direta; Pipecat (BSD-2) e LiveKit Agents (Apache 2.0) são orquestradores de nível produção; Wyoming/Rhasspy é o protocolo modular por excelência.

5. **Controle do PC deve ser modular e "permission-gated".** MCP (doado pela Anthropic em 9/12/2025 à Agentic AI Foundation, fundo dirigido sob a Linux Foundation, com membros Platinum incluindo AWS, Anthropic, Block, Bloomberg, Cloudflare, Google, Microsoft e OpenAI; suportado nativamente pelo Windows 11) é o padrão emergente; Open Interpreter, browser-use/Playwright e UI-TARS/OmniParser cobrem browser e GUI. Deixe isto para a Fase 3, sempre com confirmação humana.

## Details

### 1. Captura de áudio no Windows 11

**APIs e bibliotecas por linguagem:**
- **Python:** `sounddevice` (baseado em PortAudio, licença MIT, API limpa NumPy) e `PyAudio` são os padrões. **Importante:** o `sounddevice` upstream **não** suporta loopback WASAPI (issue aberta #281). Para capturar áudio do sistema em Python, use **PyAudioWPatch** (fork do PyAudio com PortAudio patcheado para loopback WASAPI, MIT) — tem exemplo pronto `pawp_record_wasapi_loopback.py`. Buffer de 512 amostras para baixa latência.
- **C#:** **NAudio** com `WasapiLoopbackCapture` é a referência. Gotcha documentado: se nenhum áudio toca, o evento `DataAvailable` não dispara — insira silêncio manualmente ou toque silêncio.
- **Rust:** **cpal** (multiplataforma) + crate **wasapi** (HEnquist/wasapi-rs) que já expõe `AcousticEchoCancellationControl` e tem exemplo `aec`.
- **C++:** WASAPI direto (COM).

**Loopback WASAPI:** É o mecanismo padrão para capturar o mix do sistema. A doc da Microsoft afirma: "WASAPI provides loopback mode primarily to support acoustic echo cancellation (AEC)." Requer modo shared (exclusive não suporta loopback). Formato típico: estéreo 44.1/48kHz IEEE float — você fará downmix para mono 16kHz.

**Estratégia de buffer/streaming:** Capture mic e loopback em paralelo, ambos re-amostrados para **16kHz mono** (taxa nativa do VAD, Whisper e AEC). Use frames de 10ms (160 amostras) para compatibilidade com WebRTC AEC3/VAD; blocos de 30ms para o VAD.

**VAD (detecção de atividade de voz):**
- **Silero VAD** (MIT, PyTorch/ONNX): mais preciso em ruído, deep-learning, mas adiciona latência de "várias centenas de ms" em transições fala→silêncio segundo a TEN-framework. Chunks recomendados de 150-250ms.
- **WebRTC VAD** (`py-webrtcvad`, BSD): leve, ~10-30ms frames, ótimo em detectar silêncio, mais falsos positivos em fala. Melhor para CPU mínima.
- **TEN VAD** (novo, Apache 2.0): supera Silero em precisão e complexidade, latência menor, roda em 16kHz frames de 10/16ms — vale avaliar.
- **Recomendação:** Silero VAD como padrão (qualidade), WebRTC VAD como fallback leve.

### 2. Redução de ruído / speech enhancement

- **RNNoise** (BSD, GRU-based, 22 bandas): ~10ms de latência, 1 core de CPU, leve. Excelente para chamadas WebRTC; fraco em ruído não-estacionário (ex.: TV/fala de fundo).
- **DeepFilterNet3** (MIT/Apache, deep filtering por bin): ~40ms latência, mais compute, qualidade claramente superior em ruído complexo (crowd, fala de fundo, música). Otimizado para CPU/embarcados, latência <20ms alcançável. PESQ/STOI superiores ao RNNoise.
- **noisereduce** (Python, MIT): spectral gating, bom para batch/pós-processamento, não ideal para tempo real de baixa latência.
- **SpeexDSP:** supressão de ruído clássica, integrada em muitos pipelines.
- **Krisp:** comercial (SDK pago, usado no Discord), fora do escopo local/gratuito.
- **Alternativa ao NVIDIA Broadcast/RTX Voice na AMD:** RTX Voice exige GPU NVIDIA RTX e 1-3GB VRAM — **não funciona na RX 580**. DeepFilterNet3 é o substituto open-source de qualidade equivalente rodando em CPU.
- **Recomendação:** DeepFilterNet3 no caminho do microfone (qualidade), com RNNoise como opção de baixo custo.

### 3. Wake word "Hey Elise"

- **openWakeWord** (Apache 2.0, ativamente mantido): treina wake words customizadas com **dados 100% sintéticos** (gera amostras via Piper TTS, sem gravar sua voz), exporta ONNX. Alvo: falso-aceite <0.5/hr, falso-rejeite <5%. Requer conhecimento de ML mas é o padrão open-source. **Recomendado para "Hey Elise".**
- **Picovoice Porcupine:** altíssima precisão, treino "type-to-train" em segundos, SDK Apache 2.0, roda no Windows x86_64. **Ressalva de licença:** per a licença do Porcupine (GitHub Picovoice), "Personal accounts can train custom wake word models that run on x86_64, subject to limitations and strictly for non-commercial purposes... Enterprise accounts... are permitted for use in commercial settings." O plano Foundation foi reportado em US$ 6.000 (GitHub Issue #921). AccessKey obrigatório (validação online).
- **microWakeWord** (para ESP32/embarcado — Wyoming server existe), **Vosk keyword spotting** (offline, Apache 2.0), **Mycroft Precise** (menos ativo).
- **Snowboy: descontinuado** (Kitt.AI adquirido pela Baidu, morto desde 2020) — não usar.
- **Alternativas open-source emergentes:** ViolaWake (Apache 2.0, treina TemporalCNN sobre embeddings do openWakeWord, sem AccessKey), Outspoken.
- **Estratégia always-on de baixa CPU:** wake word roda continuamente em CPU (openWakeWord ~modesto); só após o gatilho ativa-se o pipeline pesado (STT/LLM).

### 4. Distinguir SUA voz da música tocando (requisito crítico)

Este é o ponto mais difícil. A solução correta combina camadas:

**(a) AEC (Acoustic Echo Cancellation) usando loopback como referência — a base.**
- **AEC nativo do Windows 11:** a interface `IAcousticEchoCancellationControl` (audioclient.h) permite ao app definir o render endpoint usado como stream de referência via `SetEchoCancellationRenderEndpoint`. Com `NULL`, "Windows using its own algorithm to pick the loopback reference device". **Requer Windows build 22540+**. Descoberta via `IAudioClient::GetService`; se retornar `E_NOINTERFACE`, o endpoint não permite controlar o loopback de referência. **Limitação importante:** o AEC APO que usa canais privados normalmente só obtém referência do dispositivo de áudio integrado — "echo cancellation won't work if the user is playing audio out of the non-integrated device such as USB or Bluetooth". Em Windows 11 o AEC APO é inicializado com **um único** input auxiliar de referência. Sample oficial em C++: microsoft/Windows-classic-samples.
- **WebRTC AEC3** (BSD): o cancelador de eco de 3ª geração do libWebRTC (padrão no Chrome). Processa frames de **10ms a 16kHz** (160 amostras mono). Um filtro linear bem convergido atinge **20-40 dB de ERLE** (Switchboard Audio), embora em condições difíceis medições de pesquisa mostrem ~6.29 dB só no filtro linear (arXiv 2102.08551), compensado pelo supressor residual. Estágios: estimativa de atraso → filtro linear adaptativo → detector de double-talk → supressor residual.
- **speexdsp (AEC):** binding `xiongyihui/speexdsp-python` é **Pre-Alpha, Linux-only, efetivamente sem manutenção** (128 stars, 16 commits, sem wheels Windows). Frame 256, filtro 2048, mono, 16kHz. NLMS clássico — fraco em eco não-linear/música.
- **Bindings Python WebRTC recomendados para Windows:** `strands-labs/pywebrtc-audio` (pybind11, mantido, ordem HP→AEC→NS→AGC) e o pacote PyPI `aec-audio-processing` (**wheels Windows prebuilt** cp312/cp313). Em Rust: crate `aec3` (RubyBit/aec3-rs, porta do AEC3 com exemplo `karaoke_loopback.rs`).
- **Projeto de referência que já faz mic+loopback+AEC3→STT:** o plugin **Tauri 2 da SubcueAI** (Rust/cpal, "microphone + system audio (WASAPI loopback) dual capture with WebRTC AEC3 echo cancellation — 16kHz mono PCM streams for STT").

**(b) Separação de fontes (Demucs/Spleeter): NÃO em tempo real.** htdemucs é estado-da-arte mas de alta latência/compute, para uso offline. Spleeter está morto (Deezer parou em 2022). Modelos real-time de baixa latência (HS-TasNet, MMDenseNet) existem em papers mas não são plug-and-play. **Conclusão: não dependa de separação de fontes; o AEC resolve o eco do próprio PC.**

**(c) Verificação de locutor / voice fingerprint — confirma que é VOCÊ.**
- **ECAPA-TDNN** (via SpeechBrain, Apache 2.0): estado-da-arte em speaker verification, gera embedding de 192 dimensões; compare por similaridade de cosseno com seu perfil enrolado.
- **Resemblyzer** (MIT): mais simples, embeddings de locutor, bom para protótipo.
- **pyannote.audio** (MIT): diarização/verificação, mais pesado, "toolkit para construir suas ferramentas".
- **Uso:** após o wake word + AEC + VAD, extraia o embedding do trecho e só aceite o comando se bater com o dono. Isso filtra vozes vindas da música/vídeo (locutores diferentes) mesmo que passem pelo AEC.

**(d) Pipeline recomendado combinando tudo:**
`Mic (16kHz) + Loopback WASAPI (referência)` → **AEC3** (cancela áudio do PC) → **DeepFilterNet3** (ruído residual) → **Silero VAD** (detecta fala) → **openWakeWord** ("Hey Elise") → **ECAPA-TDNN** (confirma locutor) → **STT**. Beamforming/mic direcional ajuda se houver array de microfones, mas é opcional.

### 5. Speech-to-Text (STT) local

- **faster-whisper** (CTranslate2, MIT): per o README oficial SYSTRAN/faster-whisper, "This implementation is up to 4 times faster than openai/whisper for the same accuracy while using less memory. The efficiency can be further improved with 8-bit quantization on both CPU and GPU." Na AMD: rode por **DirectML** (`onnxruntime-directml`) — repo `ChharithOeun/whisper-amd-windows` demonstra faster-whisper + DirectML numa RX 5700 XT com ~8x realtime no modelo medium. Benchmark oficial (NVIDIA RTX 3070 Ti 8GB, CUDA 12.4): large-v2 int8 beam 5 = 59s, VRAM 2926MB.
- **whisper.cpp** (MIT): porta C/C++ pura, backend **Vulkan** para AMD (`-DGGML_VULKAN=ON`). O 1.8.3 traz "12x performance boost" em GPUs integradas via Vulkan. Ideal como binário único sem dependências Python. Modelos GGML (ggml-base.bin 148MB, ggml-large-v3.bin 3.1GB).
- **WhisperX:** adiciona alinhamento/diarização; **distil-whisper:** 6x mais rápido, 50% menor — **mas SOMENTE inglês** ("Distil-Whisper is currently only available for English speech recognition"). Para pt-BR, use large-v3-turbo (multilíngue) ou modelos comunitários como `freds0/distil-whisper-large-v3-ptbr` (WER 8.221% no Common Voice 16).
- **Vosk** (Apache 2.0): leve, streaming, offline, modelo pt-BR disponível, menor acurácia que Whisper. **Coqui STT:** projeto praticamente parado.
- **Português (pt-BR):** Whisper tem excelente suporte. WER na Multilingual LibriSpeech (paper Whisper, Table 10): small 13.0%, medium 9.0%, large-v2 6.8%; large-v3 melhora 10-20% sobre v2 (estimado ~5-6%). É idioma "Tier-1" quase à paridade com inglês.
- **VRAM/latência RX 580:** small (~0.5GB) e medium (~1.5GB) são o ponto ideal; large-v3 cabe mas é lento. Use streaming (RealtimeSTT alimenta chunks).
- **STT em nuvem swappable:** Deepgram, AssemblyAI, Google STT, Azure Speech — todos com pt-BR, plugáveis via a mesma interface.
- **Recomendação:** faster-whisper `small` ou `medium` (pt-BR) via DirectML como padrão; whisper.cpp Vulkan como alternativa; Deepgram/Azure como opção nuvem.

### 6. Cérebro conversacional (LLM)

**Modelos na RX 580 8GB (via llama.cpp Vulkan / LM Studio):**
- **Qwen 2.5 / Qwen3 7B-8B Q4_K_M** — **melhor escolha para pt-BR**. Qwen3 8B é apontado como o melhor LLM local para português brasileiro em 2026 (8GB VRAM, treinado em 119 idiomas, gramática pt-BR correta, forma "você"). ~15-16 tok/s.
- **Llama 3.1 8B Q4_K_M** — forte, "state-of-the-art tool use", 128K contexto, terceira melhor opção pt-BR.
- **Gemma 2/3 9B, Mistral 7B, Phi-3/4 Mini** — Mistral 7B e Phi Mini "punch above their weight". Phi-3 Mini (3.8B) ~mais rápido.
- **Sabiá-3 (Maritaca AI)** — aproxima-se de GPT-4o em português, mas não está no Ollama (download HuggingFace). Modelos pt-BR nativos: Tucano (USP), boto-9B, Amadeus-Verbo.
- **Quantização:** Q4_K_M é o ponto ideal para 8GB (7B ≈ 4.1GB). Q8 mais lento; contexto longo consome KV cache (VRAM). Comece com contexto moderado (4-8K).
- **Realismo:** modelos 14B rodam com dificuldade; 8B Q4 é o teto confortável. Use SSD (load via mmap).

**Design de backend swappable:** exponha tudo via **API OpenAI-compatible**. Ollama, LM Studio e llama.cpp `llama-server` já servem esse formato; OpenAI/Anthropic/Google/Azure são drop-in trocando base_url + api_key. Uma única camada de abstração (config-driven) permite trocar de local para nuvem por preferência.

**Memória conversacional:**
- **Curto prazo:** janela de contexto + resumo incremental (summarization) quando cresce.
- **Longo prazo:** vector DB — **Chroma** (Apache 2.0, simples), **Qdrant** (Apache 2.0, produção), ou **sqlite-vec** (leve, embarcado, ideal local). Embeddings via `nomic-embed-text` (Ollama).
- Padrão: RAG sobre memórias + resumos de sessões anteriores.

**Persona (amigo de Discord):** system prompt em pt-BR informal, respostas curtas e naturais, uso de gírias moderadas, "você"-form. Modelos edge geram respostas "mais curtas e conversacionais" — ideal para o tom.

**Function calling em modelos pequenos:** Qwen3 tem tool calling nativo no chat template (roda pela API tools do Ollama sem hacks); Llama 3.1 8B também. **Regra prática:** mantenha <5-10 ferramentas para modelos 7-8B; descrições de schema JSON são tudo; valide o JSON e limite iterações do loop (evite spinning). Per o blog oficial da Groq, o **Llama-3-Groq-8B-Tool-Use** obteve 89.06% de acurácia geral (#3 no BFCL na publicação), sendo a melhor opção open-source de 8B para 1 GPU; o 70B lidera o BFCL com 90.76% (#1).

### 7. Text-to-Speech (TTS) local com pt-BR

- **Piper** (MIT, Rhasspy): VITS ~15M params ONNX, roda em CPU em tempo real. O repositório oficial rhasspy/piper-voices lista **35 idiomas** (licença MIT), e a TTS.ai confirma "over 100 voices spanning 35+ languages at real-time speed on hardware as modest as a Pi 4." **Vozes pt-BR nativas:** `pt_BR-faber-medium`, `pt_BR-edresson`, e a comunitária **Razo** (fine-tune pt-BR conversacional). ~40ms time-to-first-audio, RTF ~0.03. Voz mais "sintética" mas prática. **Recomendado como padrão pt-BR.**
- **Kokoro** (82M, Apache 2.0): melhor qualidade/tamanho, latência mais baixa entre todos (28ms first-audio em GPU top), <0.3s processamento. **Mas sem clonagem** e suporte pt-BR limitado (foco inglês) — verificar vozes.
- **XTTS-v2** (Coqui): clonagem de voz zero-shot (3s de referência), 17 idiomas incl. pt-BR, qualidade alta. **Ressalva de licença: CPML (Coqui Public Model License) não-comercial**; Coqui fechou em 2024 (modelo sem manutenção oficial). ~600ms first-audio — difícil para tempo real.
- **F5-TTS, MeloTTS** (MIT, CPU-friendly, tempo real), **edge-tts** (vozes de nuvem grátis da Microsoft Edge, pt-BR excelentes como `pt-BR-FranciscaNeural`/`AntonioNeural` — mas é nuvem), **Windows SAPI/OneCore** (offline, qualidade inferior).
- **Streaming TTS:** **RealtimeTTS** (KoljaB, MIT) faz streaming por sentença, cai entre engines (fallback), ideal para baixa latência com saída de LLM.
- **Recomendação:** Piper (pt-BR faber/Razo) como padrão local; edge-tts como opção nuvem grátis de alta qualidade; XTTS-v2 só se quiser clonagem e uso não-comercial.

### 8. Experiência full-duplex

- **Barge-in / interrupção:** o usuário fala por cima da Elise → o VAD detecta fala e o TTS para imediatamente (padrão do Pipecat e RealtimeVoiceChat). Requer AEC também da própria voz da Elise (mesmo mecanismo de loopback).
- **Turn-taking:** detecção semântica de fim de turno (LiveKit tem turn detector transformer; Pipecat usa Silero VAD + Smart Turn model). RealtimeVoiceChat tem `turndetect.py` com detecção dinâmica de silêncio.
- **Orçamento de latência (<1-1.5s alvo):** difícil na RX 580 — realista ~1.5-2.5s. Otimize com STT streaming, LLM streaming (primeiros tokens rápido), e TTS por sentença (fala começa antes do LLM terminar).
- **Projetos para estudar/fundamentar:**
  - **RealtimeVoiceChat + RealtimeSTT + RealtimeTTS (KoljaB, MIT)** — **a base mais próxima do objetivo**: local, Ollama default, faster-whisper + Coqui/outros, barge-in, turn detection, WebSocket. **Recomendado como fundação.** Ver também Linguflex (mesmo autor, controle de ambiente por voz).
  - **Pipecat** (BSD-2, Daily): orquestrador pipeline de frames, rico em plugins, barge-in nativo. **Melhor framework de orquestração.**
  - **LiveKit Agents** (Apache 2.0): WebRTC + turn detection semântico, escala/telefonia.
  - **Wyoming/Rhasspy 3** (protocolo modular mic→wake→asr→intent→tts→snd) e **Home Assistant Assist** — excelente inspiração de arquitetura modular por domínios.
  - **OpenVoiceOS/Neon/Mycroft**, **Leon AI**, **Willow** (ESP32), **June** (CLI local Ollama+Whisper+Coqui, simples), **GLaDOS projects** (persona + Piper), **Open Interpreter** (execução de código).
  - **TEN Framework** (multimodal, low-level).

### 9. Roadmap futuro — controle do computador (fase "Jarvis" agêntica)

- **MCP (Model Context Protocol):** padrão aberto da Anthropic (nov/2024). Em 9/12/2025, a Anthropic o doou à **Agentic AI Foundation (AAIF)**, fundo dirigido sob a Linux Foundation (per David Soria Parra: "Anthropic is donating MCP to the Agentic AI Foundation, a directed fund under the Linux Foundation"), com membros Platinum incluindo AWS, Anthropic, Block, Bloomberg, Cloudflare, Google, Microsoft e OpenAI. Suportado **nativamente pelo Windows 11** (preview no Build 2025, com MCP Registry e Windows On-device Agent Registry). É o padrão para expor ferramentas ("USB-C para IA"). **Base recomendada para tools.**
- **Open Interpreter:** executa código/comandos localmente a partir de linguagem natural.
- **Automação de browser:** **Playwright** (robusto, DOM-based), **Selenium**, **browser-use** (agente LLM sobre browser).
- **Automação de UI Windows:** **pywinauto** (UI Automation API), **Windows UI Automation API** direto, **AutoHotkey** (scripts).
- **Agentes de visão/tela:** **OmniParser V2** (Microsoft, MIT — parseia screenshots em elementos estruturados, "turning any LLM into a computer use agent", SOTA no Windows Agent Arena), **UI-TARS** (ByteDance, Apache 2.0, VLM 2B/7B/72B, controle por visão pura, roda local), conceitos do **Claude computer use**.
- **Segurança/sandboxing/permissões:** **tudo permission-gated com confirmação humana** antes de ações destrutivas; isolamento de prompt, validação dual-LLM (abordagem da Microsoft), lista de allow/deny, logging de trajetória, execução em VM/sandbox quando possível. **Por que modular:** o controle de PC é o maior risco de segurança; deve ser opt-in, plugável e desligável.

### 10. Arquitetura modular recomendada

**Diagrama (em texto):**
```
┌─────────────────────────────────────────────────────────────┐
│                      BARRAMENTO DE EVENTOS                    │
│              (message bus: ZeroMQ / Redis pub-sub /           │
│               ou asyncio in-process; Wyoming-like)            │
└───┬────────┬────────┬────────┬────────┬────────┬─────────────┘
    │        │        │        │        │        │
┌───▼───┐┌───▼───┐┌───▼────┐┌──▼───┐┌───▼───┐┌───▼────┐
│ ÁUDIO ││ WAKE  ││  STT   ││ LLM  ││  TTS  ││ TOOLS  │
│Service││ WORD  ││Service ││Brain ││Service││(Fase 3)│
│(Rust/ ││openWW ││faster- ││llama.││Piper/ ││ MCP/   │
│ C++)  ││       ││whisper ││cpp   ││Kokoro ││browser │
│AEC3+  ││       ││DirectML││Vulkan││stream ││-use    │
│DFNet+ ││       ││        ││OpenAI││       ││        │
│VAD+   ││       ││        ││-compat│       ││        │
│ECAPA  ││       ││        ││abstr.││       ││        │
└───────┘└───────┘└────────┘└──────┘└───────┘└────────┘
    │                                              │
┌───▼──────────────────────────────────────────────▼──────────┐
│  MEMÓRIA (sqlite-vec/Chroma) + CONFIG (YAML: troca de modelos)│
└──────────────────────────────────────────────────────────────┘
```

**Princípios:**
- **Serviço de áudio em Rust ou C++** (baixa latência, AEC3/loopback nativo — ex.: cpal + crate `aec3` + wasapi-rs). Demais serviços em **Python** (ecossistema ML). Comunicação por barramento de eventos.
- **Sistema de plugins config-driven:** um `config.yaml` define qual backend cada serviço usa (`stt: faster-whisper | deepgram`, `llm: ollama | openai`, `tts: piper | edge-tts`). Troca sem recompilar.
- **Abstração OpenAI-compatible** central para o LLM.

**Fases/milestones:**
- **Fase 1 — MVP conversacional:** áudio (mic simples) → wake word → STT → LLM (Ollama/llama.cpp) → TTS (Piper). Sem música/AEC ainda. Meta: conversa funcional em pt-BR.
- **Fase 2 — memória, persona & robustez de áudio:** adiciona AEC3 + loopback (falar sobre música), DeepFilterNet, verificação de locutor ECAPA, memória vetorial, persona, barge-in/full-duplex.
- **Fase 3 — ferramentas & controle do PC:** function calling → MCP servers → browser-use/Playwright → UI automation → visão (OmniParser/UI-TARS), sempre permission-gated.

**Expectativas de performance realistas (RX 580 8GB):** LLM 7-8B Q4 ~15-16 tok/s; STT small/medium ~poucos segundos por enunciado (ou streaming); TTS Piper ~tempo real. Latência ponta-a-ponta ~1.5-2.5s. **Não** espere os <1s de setups NVIDIA. Considere Linux (Mesa RADV) se quiser ~2x mais desempenho Vulkan que os drivers Windows do Polaris.

**Pitfalls a evitar:**
- Tentar ROCm no Polaris (não funciona) ou Ollama+Docker GPU (falha) — use llama.cpp Vulkan.
- Não aplicar flags otimizadas para CUDA no backend Vulkan (ex.: `--override-tensor exps=CPU` degrada).
- Depender de separação de fontes em tempo real (inviável) — use AEC.
- Esquecer que o AEC nativo do Windows pode não funcionar com saída USB/Bluetooth (só integrado).
- Usar Snowboy (morto) ou XTTS-v2 comercialmente (licença não-comercial).
- WSL2 não expõe a RX 580 para Vulkan compute — compile/rode no Windows nativo.

## Recommendations

**Comece agora (Fase 1, próximas 2-4 semanas):**
1. Compile **llama.cpp com Vulkan** (`-DGGML_VULKAN=ON`) e rode `llama-server` com **Qwen3 8B Q4_K_M** (ou puxe via LM Studio, que suporta Vulkan). Valide `vulkaninfo` lista a RX 580.
2. Monte o loop conversacional sobre o **RealtimeVoiceChat (KoljaB)** como esqueleto: RealtimeSTT (faster-whisper small pt-BR) + Ollama/llama.cpp + RealtimeTTS (Piper `pt_BR-faber-medium`).
3. Treine o wake word "Hey Elise" no **openWakeWord** com dados sintéticos (Piper).

**Fase 2 (áudio robusto + memória):**
4. Implemente o serviço de áudio em Rust (cpal + wasapi-rs + crate `aec3`) OU em Python com `aec-audio-processing` (wheels Windows) + PyAudioWPatch (loopback). Cadeia: AEC3 → DeepFilterNet3 → Silero VAD → ECAPA-TDNN.
5. Adicione memória com **sqlite-vec** ou Chroma + resumo incremental.

**Fase 3 (agêntico, permission-gated):**
6. Introduza function calling (Qwen3 tools API) → **MCP** (aproveitando suporte nativo do Windows 11) → browser-use/Playwright → OmniParser/UI-TARS para GUI. Toda ação com confirmação.

**Benchmarks/limiares que mudam decisões:**
- Se LLM 8B render <8 tok/s: reduza para Phi/Qwen 3-4B ou migre para Linux/RADV.
- Se latência STT >3s: troque para whisper.cpp Vulkan ou modelo `small`/streaming.
- Se AEC nativo do Windows retornar `E_NOINTERFACE` ou você usar fone USB/BT: use WebRTC AEC3 em software com loopback explícito.
- Se pt-BR do modelo local decepcionar: teste Sabiá-3 ou caia para nuvem (config swap).

## Caveats
- **Números de tokens/s e latência variam muito** com driver (Windows AMD vs Mesa RADV no Linux — até 2x), quantização, contexto e offload. Os ~15-16 tok/s são de guias comunitários da RX 580, não medições nossas.
- **WER pt-BR do Whisper large-v3 não é publicado como número único limpo**; ~5-6% é estimativa derivada do large-v2 (6.8% MLS) + redução declarada de 10-20%.
- **ERLE do AEC3** de 20-40 dB é para filtro bem convergido (fonte comercial Switchboard); em condições difíceis medições caem a ~6 dB no filtro linear. Nenhum benchmark de ERLE específico para *música* como referência foi localizado — teste no seu setup.
- **Licenças:** Piper/Kokoro/openWakeWord/whisper.cpp/faster-whisper/RealtimeSTT-TTS/Pipecat/LiveKit são permissivas (MIT/Apache/BSD). **XTTS-v2 é não-comercial (CPML)**; **Porcupine Personal é não-comercial**. Verifique antes de qualquer uso comercial.
- **speexdsp-python está sem manutenção e é Linux-only**; prefira os bindings WebRTC (`pywebrtc-audio`, `aec-audio-processing`) ou Rust (`aec3`).
- O suporte pt-BR do **Kokoro** deve ser verificado (foco original em inglês); Piper e XTTS têm pt-BR confirmado.
- Datas/versões de projetos refletem 2024-2026; o ecossistema muda rápido — revalide repos e licenças antes de commitar arquitetura.