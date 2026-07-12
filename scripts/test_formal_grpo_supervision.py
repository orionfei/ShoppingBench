import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_console_metric_parser_and_summary():
    module = load_module("formal_supervisor", "scripts/supervise_step108_outcome_grpo.py")
    row = module.parse_step_record(
        "\x1b[36m(TaskRunner pid=1)\x1b[0m [step 5] actor/entropy=1.25, "
        "actor/pg_clipfrac=0.03, critic/score/mean=0.4, "
        "val-aux/shoppingbench_query/terminal_asr/mean@8=0.45"
    )
    assert row is not None
    assert row["step"] == 5
    assert row["actor/entropy"] == 1.25
    assert row["critic/score/mean"] == 0.4
    assert "val_terminal=0.45" in module.progress_summary(row)


def test_non_metric_line_is_ignored():
    module = load_module("formal_supervisor_nonmetric", "scripts/supervise_step108_outcome_grpo.py")
    assert module.parse_step_record("ordinary trainer output") is None


def test_formal_fallback_memory_settings_are_locked():
    launcher = (ROOT / "scripts/run_step108_outcome_grpo_formal.sh").read_text()
    assert '16) STEPS_PER_EPOCH=40; export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"' in launcher
    assert "export ROLLOUT_FREE_CACHE_ENGINE=True" in launcher
    assert 'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"' in launcher


def test_health_gate_uses_registered_limits():
    module = load_module("formal_analysis", "scripts/analyze_step108_outcome_grpo.py")
    healthy = {
        "format_mean": 0.99,
        "infrastructure_failure_rate": 0.01,
        "token_limit_noncompletion_rate": 0.08,
    }
    assert module.health(healthy, baseline_truncation=0.055)[0]
    bad = dict(healthy, infrastructure_failure_rate=0.02)
    eligible, reasons = module.health(bad, baseline_truncation=0.055)
    assert not eligible
    assert "infrastructure_failure_above_0.01" in reasons
