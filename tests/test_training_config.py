"""Configuration compatibility and independent alpha/focal switches."""
import importlib.util
from pathlib import Path

import pytest

from label_focused.training import ExperimentConfig, ensure_new_run


def cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("training_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("focal,alpha,expected", [(True, None, True), (False, None, False), (False, True, True), (True, False, False)])
def test_alpha_is_independent_with_legacy_default(focal, alpha, expected):
    config = ExperimentConfig(use_focal_loss=focal, use_class_alpha=alpha)
    assert config.class_alpha_enabled is expected
    assert config.loss_gamma == (2.0 if focal else 0.0)


@pytest.mark.parametrize("mask,focal,method", [(False, False, "base"), (True, False, "masking"), (True, True, "full")])
def test_historical_output_names_are_preserved(mask, focal, method):
    for alpha in (None, focal):
        config = ExperimentConfig(dataset="uit_vsfc", lora_rank=8, use_masking=mask, use_focal_loss=focal, use_class_alpha=alpha)
        assert config.output_dir == f"outputs/uit_vsfc/joint/{method}_r8"


def test_revision_output_names_identify_scope_seed_and_alpha():
    config = ExperimentConfig(protocol="revision", use_focal_loss=False, use_class_alpha=True)
    assert "ce_alpha_terminal_only" in config.output_dir
    assert "seed42" in config.output_dir
    other = ExperimentConfig(protocol="revision", use_focal_loss=False, use_class_alpha=True, alpha_scope="all_label_tokens", seed=43)
    assert config.output_dir != other.output_dir
    assert "all_label_tokens" in other.output_dir and "seed43" in other.output_dir


def test_revision_schedule_does_not_relax_paper_policy():
    validate = cli_module().validate_config
    validate(ExperimentConfig(protocol="revision", dataset="neu_esc", lora_rank=8, max_length=2024, max_steps=2800))
    with pytest.raises(ValueError, match="3 complete epochs"):
        validate(ExperimentConfig(max_steps=2800))
    with pytest.raises(ValueError, match="exactly 3"):
        validate(ExperimentConfig(epochs=4))
    with pytest.raises(ValueError, match="use_class_alpha"):
        validate(ExperimentConfig(use_class_alpha="false"))
    with pytest.raises(ValueError, match="alpha_scope"):
        validate(ExperimentConfig(alpha_scope="token_id"))


def test_layered_explicit_alpha_and_scope(tmp_path):
    cli = cli_module()
    dataset = tmp_path / "dataset.yaml"
    experiment = tmp_path / "experiment.yaml"
    dataset.write_text("dataset: uit_vsfc\ndefault_rank: 8\nmax_length: 1024\n")
    experiment.write_text("architecture: joint\nloss_type: cross_entropy\ncompletion_only_masking: true\nuse_class_alpha: true\nalpha_scope: all_label_tokens\nprotocol: revision\n")
    config, _ = cli.read_layered_configs(dataset, experiment)
    assert config.class_alpha_enabled and config.loss_gamma == 0.0
    assert config.alpha_scope == "all_label_tokens" and config.protocol == "revision"


def test_existing_runs_cannot_be_overwritten(tmp_path):
    run = ensure_new_run(tmp_path / "run")
    (run / "training_manifest.json").write_text("{}")
    with pytest.raises(FileExistsError, match="overwrite"):
        ensure_new_run(run)
