"""Protocol tests: corrected extra runs must not rewrite the paper matrix."""

import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_experiment(path):
    return yaml.safe_load(path.read_text())["experiment"]


def test_historical_preset_matrix_preserves_protocol_and_method_meaning():
    ranks = {"neu_esc": 32, "uit_vsfc": 8, "victsd": 8}
    expected = {}
    for dataset, rank in ranks.items():
        for method in ("base", "masking", "full"):
            expected[f"ablation/{dataset}_{method}_r{rank}.yaml"] = (
                dataset, "joint", rank, method
            )
    for dataset in ("neu_esc", "uit_vsfc"):
        for rank in (8, 16, 32, 64):
            for method in ("base", "full"):
                expected[f"joint_rank/{dataset}_{method}_r{rank}.yaml"] = (
                    dataset, "joint", rank, method
                )
    pairs = ((8, 8), (16, 16), (32, 32), (64, 64), (16, 64), (64, 16), (32, 8))
    for sent_rank, topic_rank in pairs:
        name = f"dual_rank/neu_esc_full_r{sent_rank}_r{topic_rank}.yaml"
        expected[name] = ("neu_esc", "dual_adapter", sent_rank, "full")
        config = read_experiment(ROOT / "configs" / name)
        assert config["sentiment_lora_rank"] == sent_rank
        assert config["topic_lora_rank"] == topic_rank

    actual = {
        str(path.relative_to(ROOT / "configs"))
        for folder in ("ablation", "joint_rank", "dual_rank")
        for path in (ROOT / "configs" / folder).glob("*.yaml")
    }
    assert actual == set(expected)
    for name, (dataset, architecture, rank, method) in expected.items():
        config = read_experiment(ROOT / "configs" / name)
        assert (config["dataset"], config["architecture"], config["lora_rank"]) == (
            dataset, architecture, rank
        )
        assert config["protocol"] == "paper"
        assert config["max_length"] == 1024
        assert config["epochs"] == 3 and config["max_steps"] == -1
        assert config["use_masking"] is (method != "base")
        assert config["use_focal_loss"] is (method == "full")
        assert config["use_class_alpha"] is (method == "full")
        assert config["alpha_scope"] == "terminal_only"


def test_additional_runs_separate_scope_from_focal_and_keep_recent_neu_schedule():
    paths = list((ROOT / "configs" / "additional").glob("*.yaml"))
    assert len(paths) == 10
    for dataset in ("neu_esc", "uit_vsfc", "victsd"):
        length = 2024 if dataset == "neu_esc" else 1024
        for scope in ("terminal_only", "all_label_tokens"):
            path = ROOT / "configs" / "additional" / (
                f"{dataset}_mask_alpha_ce_{scope}_r8_len{length}.yaml"
            )
            config = read_experiment(path)
            assert config["protocol"] == "revision"
            assert config["lora_rank"] == 8 and config["seed"] == 42
            assert config["use_masking"] and config["use_class_alpha"]
            assert not config["use_focal_loss"] and config["focal_gamma"] == 0
            assert config["alpha_scope"] == scope
            assert config["max_length"] == length
            assert config["max_steps"] == (2800 if dataset == "neu_esc" else -1)
        focal = read_experiment(ROOT / "configs" / "additional" / (
            f"{dataset}_mask_focal_no_alpha_r8_len{length}.yaml"
        ))
        assert focal["use_masking"] and focal["use_focal_loss"]
        assert not focal["use_class_alpha"] and focal["focal_gamma"] == 2
    base = read_experiment(ROOT / "configs" / "additional" / "victsd_base_ce_r8_len1024.yaml")
    assert not any(base[key] for key in ("use_masking", "use_focal_loss", "use_class_alpha"))


def test_layered_ce_alpha_and_focal_without_alpha_are_explicit():
    folder = ROOT / "configs" / "experiments"
    ce = yaml.safe_load((folder / "joint_mask_alpha_ce.yaml").read_text())
    assert ce["loss_type"] == "cross_entropy" and ce["use_class_alpha"]
    assert ce["alpha_scope"] == "terminal_only"
    all_tokens = yaml.safe_load((folder / "joint_mask_alpha_ce_all_label_tokens.yaml").read_text())
    assert all_tokens["loss_type"] == "cross_entropy" and all_tokens["use_class_alpha"]
    assert all_tokens["alpha_scope"] == "all_label_tokens"
    focal = yaml.safe_load((folder / "joint_mask_focal_no_alpha.yaml").read_text())
    assert focal["loss_type"] == "focal" and not focal["use_class_alpha"]


def test_config_generator_is_idempotent_and_matches_checked_in_presets(tmp_path, monkeypatch):
    generator = load_script("build_configs")
    monkeypatch.setattr(generator, "CONFIG_ROOT", tmp_path)
    generator.main()
    first = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*.yaml")}
    generator.main()
    second = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*.yaml")}
    assert first == second
    assert len(first) == 42  # 32 historical runs plus 10 opt-in corrected runs
    for name, contents in first.items():
        assert (ROOT / "configs" / name).read_bytes() == contents


def test_quickstart_is_clean_deterministic_and_has_explicit_alpha_choices():
    generator = load_script("build_notebook")
    notebook = generator.build_notebook()
    assert generator.build_notebook() == notebook
    assert json.loads((ROOT / "notebooks" / "colab_quickstart.ipynb").read_text()) == notebook
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        assert not cell["metadata"]
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'ALPHA_SCOPE = "terminal_only"' in source
    assert "all_label_tokens" in source and "use_class_alpha" in source
    assert "RUN_FULL_TRAINING = False" in source
    assert "2800" in source and "Historical paper results are archived" in source
