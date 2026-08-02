# Modo Interação Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Com wake word habilitado, cada ativação por "Hey Elise" responde só uma pergunta e volta a dormir; comandos de voz "modo interação" / "parar modo interação" alternam pra conversa contínua (comportamento atual, várias perguntas por ativação).

**Architecture:** Tudo dentro de `src/elise/orchestrator.py`. Um novo campo `Orchestrator._modo_interacao: bool` guarda o estado. `_run_turn` intercepta as duas frases-gatilho (normalizadas) antes de chamar o LLM, alterna o flag e fala uma confirmação curta direto via TTS/player. O `finally` de `_run_turn` decide entre `IDLE` e `LISTENING` conforme uma tabela de condições (wake word ligado? transcrição válida? modo interação ativo?). O timeout de inatividade em `run()` zera o flag ao dormir.

**Tech Stack:** Python 3.12, asyncio, pytest + pytest-asyncio (`asyncio_mode = auto`).

## Global Constraints

- Frases-gatilho reconhecidas apenas com `wake_enabled=True`; sem wake word, comportamento idêntico ao atual (spec, seção "Comandos de voz").
- Reconhecimento exige o texto transcrito inteiro (normalizado — minúsculo, sem acento, sem pontuação) ser exatamente `"modo interacao"` ou `"parar modo interacao"`; texto extra ao redor não dispara o comando (spec, seção "Comandos de voz").
- Comando reconhecido não entra no histórico de conversa nem invoca `self._llm.stream_reply` (spec, seção "Comandos de voz").
- Confirmações faladas, texto exato: `"Modo interação ativado."` / `"Modo interação desativado."` (spec, seção "Comandos de voz").
- Repetir um comando que já reflete o estado atual é no-op silencioso — sem confirmação nova, sem mudança de estado (spec, seção "Comandos de voz").
- Transcrição vazia nunca muda o comportamento existente — turno inválido sempre volta pra `LISTENING`, nunca `IDLE` (spec, tabela "Transições de estado").
- Timeout de inatividade em `run()` zera `_modo_interacao` ao voltar pra `IDLE` — todo novo `WakeWordDetected` começa em modo padrão (spec, seção "Transições de estado").
- Todo código, log e comentário em português, seguindo o padrão do arquivo (CLAUDE.md do projeto).

---

## Arquivo tocado

- Modify: `src/elise/orchestrator.py` — estado `_modo_interacao`, normalização/detecção de comando, helper de confirmação falada, lógica de transição de estado no fim do turno, reset no timeout.
- Test: `tests/test_orchestrator_states.py` — expande os fakes existentes (`FakeLlm`, `FakeTts`) com rastreamento de chamadas e adiciona os novos casos.

Nenhum arquivo novo — a mudança é toda no orquestrador existente, que já concentra a máquina de estados do turno.

---

### Task 1: Estado `_modo_interacao` e transição single-turn → IDLE

**Files:**
- Modify: `src/elise/orchestrator.py:74` (`__init__`), `src/elise/orchestrator.py:170-246` (`_run_turn`)
- Test: `tests/test_orchestrator_states.py`

**Interfaces:**
- Produces: `Orchestrator._modo_interacao: bool` (privado, default `False`) — lido/escrito pelas tasks seguintes.
- Produces: variável local `turno_valido: bool` dentro de `_run_turn`, controlando a decisão de estado no `finally`.

Este task ainda não implementa os comandos de voz (isso é o Task 2) — só o campo de estado e a regra de transição, testável setando `orch._modo_interacao` diretamente.

- [ ] **Step 1: Escrever testes que falham**

Editar `tests/test_orchestrator_states.py`. Primeiro, trocar `FakeLlm` e `FakeTts` por versões que registram chamadas (necessário pros testes deste task e dos seguintes) e dar a `make_orchestrator` a opção de receber um `FakeLlm` customizado:

```python
class FakeLlm:
    def __init__(self, reply_tokens: list[str] | None = None) -> None:
        self._reply_tokens = reply_tokens or []
        self.chamadas: list[str] = []

    async def stream_reply(self, user_text: str):
        self.chamadas.append(user_text)
        for tok in self._reply_tokens:
            yield tok

    def commit_reply(self, text: str) -> None:
        pass

    def rollback_user_turn(self) -> None:
        pass


class FakeTts:
    sample_rate = 16000

    def __init__(self) -> None:
        self.textos: list[str] = []

    async def synthesize(self, text: str):
        self.textos.append(text)
        return
        yield  # pragma: no cover
```

Essas duas classes substituem as versões atuais (mesmo nome, mesma posição no arquivo). Em seguida, atualizar `make_orchestrator` pra aceitar um `llm` customizado (default `FakeLlm()`, preservando o comportamento atual pros testes existentes):

```python
def make_orchestrator(
    wake_enabled: bool,
    inactivity_timeout_s: float = 45.0,
    stt_text: str = "",
    stt_delay: float = 0.0,
    llm: FakeLlm | None = None,
):
    bus = EventBus()
    cfg = EliseConfig()
    cfg.wakeword.inactivity_timeout_s = inactivity_timeout_s
    orch = Orchestrator(
        cfg,
        bus,
        FakeDenoiser(),
        FakeStt(stt_text, stt_delay),
        llm if llm is not None else FakeLlm(),
        FakeTts(),
        FakePlayer(),
        wake_enabled=wake_enabled,
    )
    return orch, bus
```

Adicionar ao final do arquivo:

```python
@pytest.mark.asyncio
async def test_wake_habilitado_dorme_apos_responder_turno_valido():
    llm = FakeLlm(reply_tokens=["Oi, tudo bem por aqui."])
    orch, bus = make_orchestrator(wake_enabled=True, stt_text="oi tudo bem", llm=llm)
    task = asyncio.create_task(orch.run())
    try:
        bus.publish(WakeWordDetected(timestamp=0.0, score=0.9, model="hey_jarvis"))
        await asyncio.sleep(0.05)
        assert orch.state is State.LISTENING

        bus.publish(UtteranceCaptured(audio=np.zeros(10, dtype=np.float32), duration_s=0.1))
        await asyncio.sleep(0.1)
        assert orch.state is State.IDLE
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_modo_interacao_ativo_continua_listening_apos_turno():
    llm = FakeLlm(reply_tokens=["Oi, tudo bem por aqui."])
    orch, bus = make_orchestrator(wake_enabled=True, stt_text="oi tudo bem", llm=llm)
    orch._modo_interacao = True
    task = asyncio.create_task(orch.run())
    try:
        bus.publish(WakeWordDetected(timestamp=0.0, score=0.9, model="hey_jarvis"))
        await asyncio.sleep(0.05)

        bus.publish(UtteranceCaptured(audio=np.zeros(10, dtype=np.float32), duration_s=0.1))
        await asyncio.sleep(0.1)
        assert orch.state is State.LISTENING
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
```

- [ ] **Step 2: Rodar os testes e confirmar que falham**

Run: `pytest tests/test_orchestrator_states.py -v`
Expected: `test_wake_habilitado_dorme_apos_responder_turno_valido` FAIL — estado fica `LISTENING`, não `IDLE` (comportamento ainda não implementado). Os testes já existentes continuam passando (fakes atualizados são compatíveis).

- [ ] **Step 3: Implementar a mudança em `orchestrator.py`**

Em `__init__`, logo após `self._wake_enabled = wake_enabled` (linha 74):

```python
        self._wake_enabled = wake_enabled
        self._modo_interacao = False
```

Em `_run_turn`, adicionar a variável `turno_valido` e trocar a checagem de transcrição vazia e o `finally`:

```python
    async def _run_turn(self, utt: UtteranceCaptured) -> None:
        spoken_parts: list[str] = []
        committed = False
        turno_valido = False
        t0 = time.monotonic()
        try:
            self._set_state(State.THINKING)

            # 1) Denoise do enunciado (CPU-bound → executor)
            loop = asyncio.get_running_loop()
            clean = await loop.run_in_executor(
                None, self._denoiser.enhance, utt.audio, self._cfg.audio.sample_rate
            )

            # 2) STT
            text = await self._stt.transcribe(clean, self._cfg.audio.sample_rate)
            t_stt = time.monotonic() - t0
            if not text or len(text.strip()) < 2:
                log.info("turno.transcricao_vazia", latencia_stt_s=round(t_stt, 2))
                return
            turno_valido = True
            log.info("usuario", texto=text, latencia_stt_s=round(t_stt, 2))
```

(o resto do bloco `try` — streaming do LLM, produtor de TTS, playback, `commit_reply` — fica igual, sem mudanças)

E o `finally` no fim do método:

```python
        finally:
            if self._wake_enabled and turno_valido and not self._modo_interacao:
                self._set_state(State.IDLE)
            else:
                self._set_state(State.LISTENING)
```

Isso substitui o antigo `self._set_state(State.LISTENING)` único do `finally`, e remove o `self._set_state(State.LISTENING)` que estava dentro do bloco de transcrição vazia (agora redundante — `turno_valido` continua `False` nesse caminho, então o `finally` já resolve pra `LISTENING`).

- [ ] **Step 4: Rodar os testes e confirmar que passam**

Run: `pytest tests/test_orchestrator_states.py -v`
Expected: PASS em todos, incluindo os dois novos e os pré-existentes (`test_wake_enabled_inicia_em_idle_com_gate_fechado`, `test_wake_desabilitado_inicia_em_listening`, `test_wakeword_detected_acorda_para_listening`, `test_timeout_sem_turno_volta_a_idle`, `test_timeout_nao_interrompe_turno_em_andamento`, `test_wake_desabilitado_nunca_entra_idle`).

- [ ] **Step 5: Rodar ruff e mypy**

Run: `ruff check src tests && mypy src`
Expected: sem erros novos.

- [ ] **Step 6: Commit**

```bash
git add src/elise/orchestrator.py tests/test_orchestrator_states.py
git commit -m "feat: single-turn por ativação de wake word, dorme após responder"
```

---

### Task 2: Comandos de voz "modo interação" / "parar modo interação"

**Files:**
- Modify: `src/elise/orchestrator.py` (imports, constantes de módulo, `_run_turn`, novo método `_falar_confirmacao`, docstring do módulo)
- Test: `tests/test_orchestrator_states.py`

**Interfaces:**
- Consumes: `Orchestrator._modo_interacao` (Task 1), tabela de transição no `finally` de `_run_turn` (Task 1) — já reage a qualquer mudança do flag, nenhuma alteração adicional necessária ali.
- Produces: função de módulo `_normalizar(texto: str) -> str`; constantes `FRASE_ATIVAR_INTERACAO` e `FRASE_DESATIVAR_INTERACAO`; método `Orchestrator._falar_confirmacao(texto: str) -> None`.

- [ ] **Step 1: Escrever testes que falham**

Adicionar ao final de `tests/test_orchestrator_states.py`:

```python
@pytest.mark.asyncio
async def test_comando_ativa_modo_interacao_sem_chamar_llm():
    llm = FakeLlm()
    orch, bus = make_orchestrator(wake_enabled=True, stt_text="Modo Interação", llm=llm)
    task = asyncio.create_task(orch.run())
    try:
        bus.publish(WakeWordDetected(timestamp=0.0, score=0.9, model="hey_jarvis"))
        await asyncio.sleep(0.05)

        bus.publish(UtteranceCaptured(audio=np.zeros(10, dtype=np.float32), duration_s=0.1))
        await asyncio.sleep(0.1)

        assert orch._modo_interacao is True
        assert llm.chamadas == []
        assert orch._tts.textos == ["Modo interação ativado."]
        assert orch.state is State.LISTENING
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_comando_desativa_modo_interacao_e_dorme():
    llm = FakeLlm()
    orch, bus = make_orchestrator(
        wake_enabled=True, stt_text="parar modo interação", llm=llm
    )
    orch._modo_interacao = True
    task = asyncio.create_task(orch.run())
    try:
        bus.publish(WakeWordDetected(timestamp=0.0, score=0.9, model="hey_jarvis"))
        await asyncio.sleep(0.05)

        bus.publish(UtteranceCaptured(audio=np.zeros(10, dtype=np.float32), duration_s=0.1))
        await asyncio.sleep(0.1)

        assert orch._modo_interacao is False
        assert llm.chamadas == []
        assert orch._tts.textos == ["Modo interação desativado."]
        assert orch.state is State.IDLE
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_comando_repetido_e_no_op():
    llm = FakeLlm()
    orch, bus = make_orchestrator(wake_enabled=True, stt_text="modo interação", llm=llm)
    orch._modo_interacao = True
    task = asyncio.create_task(orch.run())
    try:
        bus.publish(WakeWordDetected(timestamp=0.0, score=0.9, model="hey_jarvis"))
        await asyncio.sleep(0.05)

        bus.publish(UtteranceCaptured(audio=np.zeros(10, dtype=np.float32), duration_s=0.1))
        await asyncio.sleep(0.1)

        assert orch._modo_interacao is True
        assert llm.chamadas == []
        assert orch._tts.textos == []
        assert orch.state is State.LISTENING
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_frase_com_texto_extra_nao_ativa_comando():
    llm = FakeLlm()
    orch, bus = make_orchestrator(
        wake_enabled=True, stt_text="ativa o modo interação por favor", llm=llm
    )
    task = asyncio.create_task(orch.run())
    try:
        bus.publish(WakeWordDetected(timestamp=0.0, score=0.9, model="hey_jarvis"))
        await asyncio.sleep(0.05)

        bus.publish(UtteranceCaptured(audio=np.zeros(10, dtype=np.float32), duration_s=0.1))
        await asyncio.sleep(0.1)

        assert orch._modo_interacao is False
        assert llm.chamadas == ["ativa o modo interação por favor"]
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_comando_ignorado_sem_wake_word():
    llm = FakeLlm()
    orch, bus = make_orchestrator(wake_enabled=False, stt_text="modo interação", llm=llm)
    task = asyncio.create_task(orch.run())
    try:
        bus.publish(UtteranceCaptured(audio=np.zeros(10, dtype=np.float32), duration_s=0.1))
        await asyncio.sleep(0.1)

        assert orch._modo_interacao is False
        assert llm.chamadas == ["modo interação"]
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
```

- [ ] **Step 2: Rodar os testes e confirmar que falham**

Run: `pytest tests/test_orchestrator_states.py -v`
Expected: os 5 testes novos FAIL (comando ainda não é reconhecido — texto vai direto pro LLM em todos os casos).

- [ ] **Step 3: Implementar a detecção de comando em `orchestrator.py`**

No topo do arquivo, adicionar import e constantes (logo abaixo dos imports existentes, antes de `log = structlog.get_logger(__name__)`):

```python
import re
import unicodedata
```

(adicionar junto aos imports já existentes `import asyncio`, `import contextlib`, `import time`)

Depois de `log = structlog.get_logger(__name__)`:

```python
FRASE_ATIVAR_INTERACAO = "modo interacao"
FRASE_DESATIVAR_INTERACAO = "parar modo interacao"


def _normalizar(texto: str) -> str:
    """minúsculo, sem acento, sem pontuação — pra comparar com frase-gatilho."""
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"[^a-z0-9]+", " ", sem_acento.lower()).strip()
```

No docstring do módulo (topo do arquivo), adicionar um bullet depois do bullet sobre wake word (depois de "...pode perder as primeiras sílabas."):

```
- Modo interação: por padrão, cada ativação por wake word responde uma
  pergunta e volta a dormir (IDLE). Os comandos de voz "modo interação" e
  "parar modo interação" alternam pra conversa contínua (várias perguntas
  por ativação, como antes) e de volta ao padrão single-turn.
```

Em `_run_turn`, entre `turno_valido = True` e `log.info("usuario", ...)` (que o Task 1 deixou assim):

```python
            turno_valido = True

            if self._wake_enabled:
                normalizado = _normalizar(text)
                if normalizado == FRASE_ATIVAR_INTERACAO:
                    if not self._modo_interacao:
                        self._modo_interacao = True
                        await self._falar_confirmacao("Modo interação ativado.")
                    return
                if normalizado == FRASE_DESATIVAR_INTERACAO:
                    if self._modo_interacao:
                        self._modo_interacao = False
                        await self._falar_confirmacao("Modo interação desativado.")
                    return

            log.info("usuario", texto=text, latencia_stt_s=round(t_stt, 2))
```

Novo método privado, colocado depois de `_set_state` (antes do comentário `# Um turno completo`):

```python
    async def _falar_confirmacao(self, texto: str) -> None:
        """Fala uma frase fixa direto (sem LLM, sem histórico de conversa)."""
        chunks = [c async for c in self._tts.synthesize(texto)]
        self._set_state(State.SPEAKING)
        log.info("elise", texto=texto)
        await self._player.play(_aiter(chunks), self._tts.sample_rate)
```

- [ ] **Step 4: Rodar os testes e confirmar que passam**

Run: `pytest tests/test_orchestrator_states.py -v`
Expected: PASS em todos os testes, novos e antigos.

- [ ] **Step 5: Rodar ruff e mypy**

Run: `ruff check src tests && mypy src`
Expected: sem erros novos.

- [ ] **Step 6: Commit**

```bash
git add src/elise/orchestrator.py tests/test_orchestrator_states.py
git commit -m "feat: comandos de voz modo interação / parar modo interação"
```

---

### Task 3: Resetar `_modo_interacao` no timeout de inatividade

**Files:**
- Modify: `src/elise/orchestrator.py:127-129` (branch `except asyncio.TimeoutError` dentro de `run()`)
- Test: `tests/test_orchestrator_states.py`

**Interfaces:**
- Consumes: `Orchestrator._modo_interacao` (Task 1).

- [ ] **Step 1: Escrever teste que falha**

Adicionar ao final de `tests/test_orchestrator_states.py`:

```python
@pytest.mark.asyncio
async def test_timeout_zera_modo_interacao():
    orch, bus = make_orchestrator(wake_enabled=True, inactivity_timeout_s=0.1)
    orch._modo_interacao = True
    task = asyncio.create_task(orch.run())
    try:
        bus.publish(WakeWordDetected(timestamp=0.0, score=0.9, model="hey_jarvis"))
        await asyncio.sleep(0.05)
        assert orch.state is State.LISTENING

        await asyncio.sleep(0.15)  # > inactivity_timeout_s, sem enunciado
        assert orch.state is State.IDLE
        assert orch._modo_interacao is False
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
```

- [ ] **Step 2: Rodar o teste e confirmar que falha**

Run: `pytest tests/test_orchestrator_states.py::test_timeout_zera_modo_interacao -v`
Expected: FAIL — `orch._modo_interacao` continua `True` depois do timeout.

- [ ] **Step 3: Implementar o reset em `orchestrator.py`**

Dentro de `run()`, no branch de timeout:

```python
                    except asyncio.TimeoutError:
                        self._modo_interacao = False
                        self._set_state(State.IDLE)
                        continue
```

- [ ] **Step 4: Rodar os testes e confirmar que passam**

Run: `pytest tests/test_orchestrator_states.py -v`
Expected: PASS em todos, incluindo o suite completo (12 testes: 6 pré-existentes + 6 adicionados nas Tasks 1–3).

- [ ] **Step 5: Rodar suite completa, ruff e mypy**

Run: `pytest && ruff check src tests && mypy src`
Expected: sem erros.

- [ ] **Step 6: Commit**

```bash
git add src/elise/orchestrator.py tests/test_orchestrator_states.py
git commit -m "feat: zera modo interação ao dormir por inatividade"
```
