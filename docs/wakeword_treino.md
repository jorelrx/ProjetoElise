# Treinando o wake word "Hey Elise"

O backend de wake word (`src/elise/audio/wakeword.py`) usa o
[openWakeWord](https://github.com/dscripka/openWakeWord) (Apache 2.0). O `config.yaml`
já aponta `wakeword.model` para `models/wakeword/hey_elise.onnx` — enquanto esse
arquivo não existir, o backend detecta a ausência (`resolve_model_name` em
`wakeword.py`) e usa automaticamente `wakeword.fallback_model` (`hey_jarvis`, um
modelo pré-treinado em inglês, que serve só de placeholder pra validar a integração).
Para reconhecer "Hey Elise" de verdade é preciso treinar o modelo customizado.

## Caminho recomendado (dados sintéticos, sem gravar sua voz)

O openWakeWord permite treinar a partir de amostras 100% sintéticas geradas por TTS,
sem precisar de um dataset gravado à mão:

1. Siga o notebook oficial de treino do repositório `dscripka/openWakeWord`
   (`notebooks/training_models.ipynb`) — ele cobre geração de amostras positivas/negativas
   sintéticas, aumento de dados (ruído, reverberação) e o treino do classificador.
2. Gere as amostras positivas ("Hey Elise", variações de entonação/velocidade) usando o
   próprio **Piper** (já é dependência do projeto) com a voz pt-BR configurada em
   `tts.piper.model_path`, em vez das vozes em inglês do notebook original.
3. Alvos de qualidade sugeridos pelo estudo técnico do projeto (`docs/estudo_base.md`
   §3): falso-aceite **< 0,5/hora** e falso-rejeite **< 5%**. Valide gravando um punhado
   de sessões reais (você falando normalmente, sem querer acordar a Elise) e contando
   falsos-positivos.
4. Exporte o modelo treinado em **ONNX** (o backend do projeto usa
   `inference_framework="onnx"` — evita a dependência `tflite-runtime`, problemática no
   Windows).

## Plugando o modelo treinado

Salve o `.onnx` resultante em `models/wakeword/hey_elise.onnx` — é exatamente o caminho
que `wakeword.model` já espera no `config.yaml`; nenhuma outra mudança de config é
necessária, o fallback pra `hey_jarvis` deixa de ser usado assim que o arquivo existir.

Requer o extra instalado: `pip install -e .[wakeword]`.
