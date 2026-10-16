import json
from pathlib import Path

import pytest

from enterprise_agent_rl.inference_stability import (
    FiniteLogitsProcessor,
    NonFiniteLogitsError,
    attention_backend_context,
    check_finite_logits,
    transformers_attention_implementation,
)
from enterprise_agent_rl.model_server import ModelSettings, QwenEngine, append_event


def test_model_config_defaults_to_sdpa_math():
    config = ModelSettings.load(Path("configs/model/qwen2.5-1.5b.toml"))
    assert config.attention_backend == "sdpa_math"
    assert transformers_attention_implementation(config.attention_backend) == "sdpa"


def test_model_config_can_select_eager(tmp_path):
    config_path = tmp_path / "model.toml"
    config_path.write_text(
        'model_id = "Qwen/Qwen2.5-1.5B-Instruct"\n'
        'revision = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"\n'
        'attention_backend = "eager"\n'
    )
    assert ModelSettings.load(config_path).attention_backend == "eager"


def test_cpu_attention_context_is_a_noop():
    class CpuOnlyTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False

    with attention_backend_context(CpuOnlyTorch, "sdpa_math") as control:
        assert control["status"] == "not_applicable"
        assert control["reason"] == "cuda_unavailable"
        assert control["math"] is True


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_non_finite_logits_guard_fails_with_diagnostic_context(bad_value):
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.0, bad_value, -1.0]], dtype=torch.float32)
    processor = FiniteLogitsProcessor(
        prompt_tokens=3,
        context={
            "attention_backend": "eager",
            "model_revision": "revision-1",
            "model_dtype": "torch.float16",
            "gpu_name": "test-gpu",
            "sdpa_backend_control": {"status": "not_applicable"},
        },
    )

    with pytest.raises(NonFiniteLogitsError, match="non_finite_logits") as raised:
        processor(torch.zeros((1, 4), dtype=torch.long), logits)

    payload = raised.value.payload
    assert payload["generation_step"] == 1
    assert payload["logits_shape"] == [1, 3]
    assert payload["dtype"] == "torch.float32"
    assert payload["device"] == "cpu"
    assert payload["finite_ratio"] == pytest.approx(2 / 3)
    assert payload["attention_backend"] == "eager"
    assert payload["model_revision"] == "revision-1"
    assert payload["gpu_name"] == "test-gpu"
    assert payload["diagnostic_context"]["sdpa_backend_control"]["status"] == "not_applicable"
    assert processor.steps_checked == 1


def test_finite_logits_guard_returns_original_tensor():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.0, 1.0, -1.0]], dtype=torch.float32)
    assert (
        check_finite_logits(
            logits,
            generation_step=0,
            context={"attention_backend": "sdpa_math"},
        )
        is logits
    )


def test_telemetry_backend_is_serialized_in_artifact(tmp_path):
    engine = object.__new__(QwenEngine)
    engine.cfg = ModelSettings(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        attention_backend="sdpa_math",
    )
    engine.model_dtype = "torch.float16"
    engine.device = "cuda:0"
    engine.gpu_name = "Tesla V100S-PCIE-32GB"
    engine.compute_capability = [7, 0]
    engine.load_time = 1.25
    engine.backend_control = {
        "requested": "sdpa_math",
        "status": "forced",
        "api": "torch.backends.cuda.sdp_kernel",
        "flash": False,
        "math": True,
        "mem_efficient": False,
    }
    path = tmp_path / "generations.jsonl"
    append_event(path, {"telemetry": engine._base_telemetry()})

    artifact = json.loads(path.read_text())
    telemetry = artifact["telemetry"]
    assert telemetry["attention_backend"] == "sdpa_math"
    assert telemetry["model_revision"] == engine.cfg.revision
    assert telemetry["gpu_name"] == "Tesla V100S-PCIE-32GB"
    assert telemetry["non_finite_logits_detected"] is False
