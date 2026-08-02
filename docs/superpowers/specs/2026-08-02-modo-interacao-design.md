# Modo interação — desenho

Data: 2026-08-02

## Contexto

Hoje, com wake word habilitado (`wakeword.enabled: true`), o fluxo é:

1. `IDLE` (dormindo) → `WakeWordDetected` → `LISTENING`.
2. Enunciado capturado → `THINKING` → `SPEAKING` → volta pra `LISTENING`.
3. Em `LISTENING` sem turno ativo, `Orchestrator.run()` espera o próximo
   enunciado com timeout de `wakeword.inactivity_timeout_s` (padrão 45s).
   Se o timeout estourar, volta pra `IDLE`.

Ou seja, depois de acordar, a Elise fica numa "janela de conversa" de até
45s entre turnos antes de dormir de novo — mesmo respondendo só uma
pergunta.

## Objetivo

Ativação por wake word continua igual. Mas o padrão passa a ser
**um turno por ativação**: depois de responder, a Elise dorme na hora,
sem esperar mais nada. Um modo contínuo (o comportamento atual, várias
perguntas por ativação) só entra por comando de voz explícito.

## Comandos de voz

Dois comandos reconhecidos, só quando `wake_enabled=True`:

- `"modo interação"` → ativa o modo contínuo.
- `"parar modo interação"` → desativa, volta ao padrão single-turn.

Reconhecimento: o texto transcrito inteiro (normalizado — minúsculo, sem
acento, sem pontuação) precisa **ser exatamente** a frase-gatilho. Frases
com texto extra ao redor (ex: "ativa o modo interação por favor") não
disparam o comando — seguem como pergunta normal pro LLM.

Comando reconhecido não entra no histórico de conversa nem passa pelo LLM.
Elise responde com confirmação curta, sintetizada e tocada diretamente:

- `"Modo interação ativado."`
- `"Modo interação desativado."`

Se o comando repetir um estado que já está ativo (ex: dizer "modo
interação" já estando em modo interação), é tratado como no-op silencioso
— nenhuma confirmação nova, nenhuma mudança de estado.

Sem wake word habilitado (`wake_enabled=False`), essas frases não são
interceptadas — comportamento idêntico ao atual, tudo vai pro LLM.

## Estado

Novo campo no `Orchestrator`: `self._modo_interacao: bool = False`.
Só é lido/escrito quando `wake_enabled=True`.

## Transições de estado

Ao fim de um turno (`finally` de `_run_turn`), a lógica atual sempre volta
pra `LISTENING`. Passa a ser:

| `wake_enabled` | transcrição vazia | `modo_interacao` | próximo estado |
|---|---|---|---|
| False | — | — | `LISTENING` (sem mudança) |
| True | sim | — | `LISTENING` (sem mudança — protege contra STT perder a primeira sílaba, caso conhecido) |
| True | não | False (padrão) | `IDLE` — dorme direto após responder |
| True | não | True | `LISTENING` — continua ouvindo, como hoje |

Comandos de modo (`"modo interação"` / `"parar modo interação"`) contam
como "transcrição não vazia" pra essa tabela — depois de ativar o modo
contínuo, o próximo estado é `LISTENING` (fica ouvindo); depois de
desativar, o próximo estado é `IDLE` (dorme).

No timeout de inatividade em `Orchestrator.run()` (branch `TimeoutError`,
volta pra `IDLE`), zera também `self._modo_interacao = False`. Um novo
`WakeWordDetected` sempre começa em modo padrão (single-turn) — o modo
contínuo não sobrevive a um período de silêncio.

## Detecção do comando — onde entra no código

Dentro de `_run_turn`, logo depois do STT (`text = await self._stt.transcribe(...)`)
e da checagem de transcrição vazia já existente, antes do log `"usuario"`
e do envio pro LLM:

```python
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
```

`_normalizar` — minúsculo + remoção de acento (`unicodedata`) + strip de
pontuação/espaços extras.

`_falar_confirmacao(texto)` — sintetiza e toca a frase fixa via
`self._tts` + `self._player`, igual ao pipeline de sentença já usado no
turno normal (`_aiter`), sem passar pelo LLM nem pelo histórico de
conversa. Ajusta estado pra `SPEAKING` durante a fala; o estado final
(`IDLE`/`LISTENING`) é decidido no `finally` de `_run_turn` conforme a
tabela acima.

## Erros e casos de borda

- Comando dito durante `modo_interacao` já ativo/inativo (repetido): no-op,
  sem confirmação duplicada (evita eco de "ativado" toda hora se o STT
  pegar ruído parecido com o comando).
- Falha na síntese/playback da confirmação: mesmo tratamento de erro dos
  demais `await` no turno — propaga pra dentro do `except Exception` do
  `_run_turn`, loga `turno.erro`, não derruba o processo.
- Barge-in / `full_duplex` durante confirmação: mesma lógica de cancelamento
  de turno já existente (`_cancel_turn`), sem tratamento especial.

## Testes

- Unitário em `Orchestrator`: dado `wake_enabled=True`, `modo_interacao=False`,
  turno com texto qualquer (não-comando) → estado final `IDLE`.
- Dado `modo_interacao=True` → estado final `LISTENING` após turno normal.
- Texto exato `"modo interação"` (com variação de acento/caixa) → ativa,
  fala confirmação, não chama o LLM (`stream_reply` não invocado).
- Texto exato `"parar modo interação"` com `modo_interacao=True` → desativa,
  confirmação, estado final `IDLE`.
- Texto `"modo interação"` com `modo_interacao` já `True` → no-op, sem
  chamada de TTS de confirmação, estado final `LISTENING` (turno "válido"
  contando como conversa contínua).
- `TimeoutError` em `run()` com `modo_interacao=True` → volta `IDLE` e
  zera `modo_interacao`.
- Frase com texto extra ("ativa modo interação por favor") → não ativa,
  segue fluxo normal pro LLM.
- `wake_enabled=False` com texto `"modo interação"` → vai pro LLM normal,
  sem interceptação.
