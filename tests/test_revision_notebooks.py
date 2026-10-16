"""Release checks for controlled alpha ablations and the standalone CE baseline."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "notebooks/revisions"
MANIFEST = json.loads((REVISION / "PROVENANCE.json").read_text())


def notebook_tree(path):
    notebook = json.loads(path.read_text())
    source = "\n\n".join(
        "\n".join(line for line in "".join(cell["source"]).splitlines()
                  if not line.lstrip().startswith(("!", "%")))
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    return ast.parse(source)


def assignment(tree, name):
    values = [ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == name
                      for target in node.targets)]
    assert len(values) == 1, name
    return values[0]


@pytest.mark.parametrize("record", MANIFEST["notebooks"], ids=lambda item: Path(item["path"]).stem)
def test_published_training_notebook_is_clean_and_verifiable(record):
    path = ROOT / record["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["published_sha256"]
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
    text = path.read_text()
    assert not re.search(r"(?:hf_|gh[pousr]_)[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}", text)
    assert "/home/loc/" not in text
    tree = notebook_tree(path)
    assert assignment(tree, "SEED") == 42
    assert assignment(tree, "USE_ALPHA") is (record["alpha_scope"] is not None)
    assert assignment(tree, "FOCAL_GAMMA") == 0.0
    if record["alpha_scope"] is not None:
        assert assignment(tree, "ALPHA_SCOPE") == record["alpha_scope"]


class NormalizeScope(ast.NodeTransformer):
    def visit_Assign(self, node):
        if any(isinstance(target, ast.Name) and target.id == "ALPHA_SCOPE" for target in node.targets):
            node.value = ast.Constant("<scope>")
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value.endswith(".ipynb"):
            node.value = node.value.replace("terminal_only", "all_label_tokens")
        return node


@pytest.mark.parametrize("dataset", ["uit_vsfc", "neu_esc", "victsd"])
def test_terminal_all_pair_changes_only_alpha_scope(dataset):
    pair = [record for record in MANIFEST["notebooks"]
            if record["dataset"] == dataset and record["alpha_scope"] is not None]
    assert len(pair) == 2
    trees = [NormalizeScope().visit(notebook_tree(ROOT / record["path"])) for record in pair]
    assert ast.dump(trees[0]) == ast.dump(trees[1])


def test_standalone_base_ce_supervises_prompt_and_ignores_padding():
    tree = notebook_tree(REVISION / "victsd/victsd_base_ce_r8_len1024.ipynb")
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "full_sequence_ce_loss")
    namespace = {"F": F}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "base_ce_notebook", "exec"), namespace)
    logits = torch.tensor([[[1., 2., 0.], [3., 0., 1.], [0., 2., 1.], [1., 0., 3.]]], requires_grad=True)
    # Both the prompt target (position 1) and response target (position 2) count.
    labels = torch.tensor([[0, 1, 2, -100]])
    actual = namespace["full_sequence_ce_loss"](logits, labels)
    expected = torch.stack([-logits[0, 0].log_softmax(-1)[1], -logits[0, 1].log_softmax(-1)[2]]).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad[0, 0].abs().sum() > 0
    assert logits.grad[0, 1].abs().sum() > 0
    assert logits.grad[0, 2:].abs().sum() == 0


def test_victsd_revision_scorer_retains_invalid_field_as_error():
    path = ROOT / "scripts/revisions/victsd_test_ablation_fixed.py"
    spec = importlib.util.spec_from_file_location("victsd_revision_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    valid = module.parse_output('{"Sentiment":"Toxic","Topic":"TheGioi"}')
    partial = module.parse_output('{"Sentiment":"Toxic","Topic":null}')
    duplicate = module.parse_output('{"Sentiment":"Toxic","Sentiment":"Non-toxic","Topic":"TheGioi"}')
    assert valid["sentiment_valid"] and valid["topic_valid"]
    assert partial["pred_sentiment"] == "Toxic"
    assert partial["pred_topic"] == module.INVALID_LABEL
    assert not duplicate["schema_valid"]
    assert duplicate["pred_sentiment"] == module.INVALID_LABEL
    assert duplicate["pred_topic"] == module.INVALID_LABEL
    # Fixed-class F1 and confusion counts must include the failed generations.
    module.self_check()


def test_neu_validation_scores_invalid_outputs_as_errors_without_train_labels():
    from sklearn.metrics import f1_score

    path = ROOT / "scripts/revisions/neu_val_mf1_3_checkpoints.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_parse_val_output", "_score_val_outputs"}]
    namespace = {
        "json": json, "f1_score": f1_score, "INVALID_LABEL": "PARSE_ERROR",
        "SENTIMENT_LABELS": ["Neutral", "Positive", "Negative", "Toxic"],
        "TOPIC_LABELS": ["Spam", "News", "Academic", "Other", "Service",
                         "Jobs & Recruitment", "Personal Affairs", "Social Affairs",
                         "Help & Share", "Club & Events"],
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    raw = ["invalid JSON", '{"Sentiment":"Neutral","Topic":null}',
           '{"Sentiment":"Neutral","Topic":"News"}']
    parsed, invalid_s, invalid_t, pred_s, pred_t, mf1_s, mf1_t = namespace["_score_val_outputs"](
        raw, ["Neutral"] * 3, ["News"] * 3,
    )
    assert len(parsed) == len(raw)
    assert invalid_s == [True, False, False]
    assert invalid_t == [True, True, False]
    assert pred_s == ["PARSE_ERROR", "Neutral", "Neutral"]
    assert pred_t == ["PARSE_ERROR", "PARSE_ERROR", "News"]
    assert mf1_s == pytest.approx(0.8 / 4)
    assert mf1_t == pytest.approx(0.5 / 10)
    with pytest.raises(ValueError, match="validation"):
        namespace["_score_val_outputs"](raw[:1], ["Neutral"] * 3, ["News"] * 3)
