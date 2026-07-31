from pathlib import Path

import pytest
from pydantic import ValidationError

from elise.config import DuplexMode, EliseConfig, load_config
from elise.events import EventBus


def test_config_defaults_sem_arquivo(tmp_path: Path):
    cfg = load_config(tmp_path / "nao_existe.yaml")
    assert cfg.llm.model == "qwen3.5-9b"
    assert cfg.behavior.mode is DuplexMode.HALF
    assert cfg.audio.frame_samples == 512


def test_config_yaml_sobrescreve(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "llm:\n  model: outro-modelo\n  base_url: http://127.0.0.1:8080/v1\n"
        "behavior:\n  mode: full_duplex\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.llm.model == "outro-modelo"
    assert cfg.behavior.mode is DuplexMode.FULL


def test_sample_rate_invalido_falha_cedo():
    with pytest.raises(ValidationError):
        EliseConfig.model_validate({"audio": {"sample_rate": 44100}})


def test_bus_drop_oldest_sob_pressao():
    bus = EventBus()
    q = bus.subscribe(int, maxsize=2)
    for i in range(5):
        bus.publish(i)
    assert q.qsize() == 2
    assert q.get_nowait() == 3  # os mais antigos foram descartados
    assert q.get_nowait() == 4
