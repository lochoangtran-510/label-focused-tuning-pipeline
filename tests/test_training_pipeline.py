"""CPU integration of filtering, objective switches, and Trainer wiring."""
import json
from types import ModuleType, SimpleNamespace
import sys

import pandas as pd
import pytest
import torch

from label_focused import training
from label_focused.datasets import DATASET_REGISTRY


class Dataset:
    def __init__(self, data): self.data = data
    @classmethod
    def from_dict(cls, data): return cls(data)
    def __len__(self): return len(self.data["text"])
    def __getitem__(self, index): return {key: values[index] for key, values in self.data.items()}


class Tokenizer:
    padding_side = "right"
    pad_token_id = 2
    eos_token_id = 2
    chat_template = "deterministic test template"
    special_tokens_map = {"eos_token": "</s>"}
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) + 3 for char in text]
    def decode(self, ids, **kwargs): return "".join(chr(int(token) - 3) for token in ids)
    def get_vocab(self): return {"a": 100}
    def apply_chat_template(self, messages, **kwargs):
        return "[INST] " + "\n".join(message["content"] for message in messages[:-1]) + " [/INST] " + messages[-1]["content"]
    def __call__(self, texts, **kwargs):
        assert kwargs.get("add_special_tokens") is False
        if isinstance(texts, str):
            ids = self.encode(texts)
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return {"input_ids": [self.encode(text) for text in texts]}


class InnerCollator:
    def __init__(self, tokenizer, masked): self.tokenizer, self.masked = tokenizer, masked
    def __call__(self, features):
        width = max(len(feature["input_ids"]) for feature in features)
        ids = torch.zeros((len(features), width), dtype=torch.long)
        attention = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for row, feature in enumerate(features):
            values = feature["input_ids"]
            n = len(values)
            ids[row, :n] = torch.tensor(values)
            attention[row, :n] = 1
            text = self.tokenizer.decode(values)
            start = text.rfind("[/INST]") + len("[/INST]") if self.masked else 0
            labels[row, start:n] = ids[row, start:n]
        return {"input_ids": ids, "attention_mask": attention, "labels": labels}


@pytest.fixture
def runtime(monkeypatch):
    datasets = ModuleType("datasets"); datasets.Dataset = Dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    events = []
    transformers = ModuleType("transformers")
    transformers.set_seed = lambda seed: (torch.manual_seed(seed), events.append(("seed", seed)))
    transformers.TrainingArguments = lambda **kwargs: SimpleNamespace(**kwargs)
    transformers.EarlyStoppingCallback = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    peft = ModuleType("peft")
    peft.LoraConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    def get_peft_model(model, config):
        events.append(("lora", torch.rand(1).item())); return model
    peft.get_peft_model = get_peft_model
    monkeypatch.setitem(sys.modules, "peft", peft)
    trl = ModuleType("trl")
    class SFTTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = kwargs["model"]
            self.args = kwargs["args"]
    trl.SFTTrainer = SFTTrainer
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setattr(training, "make_completion_collator", lambda tokenizer: InnerCollator(tokenizer, True))
    monkeypatch.setattr(training, "make_full_sequence_collator", lambda tokenizer: InnerCollator(tokenizer, False))
    return events


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.parameter = torch.nn.Parameter(torch.randn(7, 7)); self.received = None
    def enable_input_require_grads(self): pass
    def forward(self, input_ids, attention_mask):
        self.received = {"input_ids": input_ids, "attention_mask": attention_mask}
        class Output(dict):
            __getattr__ = dict.__getitem__
        return Output(logits=self.parameter[input_ids])


def frame():
    rows = [{"sentence": "short", "sentiment": s, "topic": t} for s in range(3) for t in range(4)]
    rows.append({"sentence": "short", "sentiment": 2, "topic": 0})
    rows.append({"sentence": "long" * 1000, "sentiment": 0, "topic": 1})
    return pd.DataFrame(rows)


def datasets(runtime, train=None, task=None):
    train = frame() if train is None else train
    val = train.iloc[:1].copy()
    spec = DATASET_REGISTRY["uit_vsfc"]
    if task:
        return training.make_single_task_datasets(train, val, spec, Tokenizer(), 1500, task)
    return training.make_prompt_datasets(train, val, spec, Tokenizer(), 1500)


def test_exact_filtered_train_and_validation_no_truncation(runtime):
    train, validation = datasets(runtime)
    retained = training.filtered_train_frame(train)
    assert len(retained) == 13 and len(train) == 13 and len(validation) == 1
    assert not retained["sentence"].str.startswith("long").any()
    weights = training.sample_weights(retained, DATASET_REGISTRY["uit_vsfc"])
    assert weights["Sentiment"]["Negative"] == pytest.approx(13 / (3 * 4))
    assert weights["Topic"]["Lecturer"] == pytest.approx(13 / (4 * 4))
    with pytest.raises(ValueError, match="validation examples exceed"):
        training.make_prompt_datasets(frame(), frame().iloc[-1:], DATASET_REGISTRY["uit_vsfc"], Tokenizer(), 1500)
    with pytest.raises(ValueError, match="No training examples"):
        training.make_prompt_datasets(frame().iloc[-1:], frame().iloc[:1], DATASET_REGISTRY["uit_vsfc"], Tokenizer(), 1500)


@pytest.mark.parametrize("alpha,focal", [(False, False), (True, False), (False, True), (True, True)])
def test_independent_objectives_manifest_and_constructor(runtime, tmp_path, alpha, focal):
    train, validation = datasets(runtime)
    config = training.ExperimentConfig(dataset="uit_vsfc", output_root=str(tmp_path), lora_rank=8, protocol="revision", use_class_alpha=alpha, use_focal_loss=focal, alpha_scope="all_label_tokens")
    trainer = training.make_trainer(Model(), Tokenizer(), train, validation, frame(), config)
    assert trainer.model_accepts_loss_kwargs is False
    assert runtime[-2][0] == "seed" and runtime[-1][0] == "lora"
    assert trainer.args.seed == trainer.args.data_seed == 42
    assert trainer.args.label_names == ["labels"] and trainer.args.prediction_loss_only
    assert trainer.kwargs["packing"] is False
    assert trainer.kwargs["dataset_kwargs"] == {"add_special_tokens": False}
    features = [Tokenizer()(train[0]["text"], add_special_tokens=False)]
    batch = trainer.kwargs["data_collator"](features)
    assert ("alpha_weights" in batch) is alpha
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.ones(1, 4, dtype=torch.long), "labels": torch.tensor([[-100, 2, 3, 4]])}
    if alpha: inputs["alpha_weights"] = torch.tensor([[1., 2., .5, 3.]])
    loss, output = trainer.compute_loss(trainer.model, inputs, return_outputs=True, num_items_in_batch=17)
    expected = training.label_focused_loss(output.logits, inputs["labels"], 2.0 if focal else 0., inputs.get("alpha_weights"))
    torch.testing.assert_close(loss, expected)
    assert "labels" in inputs and "labels" not in trainer.model.received
    assert "alpha_weights" not in trainer.model.received
    loss.backward(); assert torch.isfinite(trainer.model.parameter.grad).all()
    manifest = json.loads((__import__("pathlib").Path(config.output_dir) / "training_manifest.json").read_text())
    assert manifest["data"]["filtered_train_examples"] == 13
    assert manifest["semantics"]["focal_gamma"] == (2.0 if focal else 0.)
    assert manifest["semantics"]["padding_mask"] == "trl_completion_mask_and_pad_id_mask"
    assert manifest["semantics"]["true_eos_supervised"] is False
    assert bool(manifest["class_weights_by_field"]) is alpha
    assert len(manifest["tokenizer_sha256"]) == 64
    with pytest.raises(FileExistsError):
        training.make_trainer(Model(), Tokenizer(), train, validation, frame(), config)


def test_missing_classes_only_block_enabled_alpha(runtime, tmp_path):
    train, validation = datasets(runtime, frame().iloc[:1])
    config = training.ExperimentConfig(dataset="uit_vsfc", output_root=str(tmp_path), protocol="revision", use_class_alpha=False)
    training.make_trainer(Model(), Tokenizer(), train, validation, frame(), config)
    config.use_class_alpha = True
    with pytest.raises(ValueError, match="missing Sentiment"):
        training.make_trainer(Model(), Tokenizer(), train, validation, frame(), config)


@pytest.mark.parametrize("task,field", [("sentiment", "Sentiment"), ("topic", "Topic")])
def test_single_task_alpha_is_one_field(runtime, tmp_path, task, field):
    train, validation = datasets(runtime, task=task)
    config = training.ExperimentConfig(dataset="uit_vsfc", output_root=str(tmp_path), protocol="revision", use_class_alpha=True, use_focal_loss=False)
    trainer = training.make_trainer(Model(), Tokenizer(), train, validation, frame(), config, task=task)
    batch = trainer.kwargs["data_collator"]([Tokenizer()(train[0]["text"], add_special_tokens=False)])
    assert "alpha_weights" in batch
    weights = trainer.kwargs["data_collator"].weights_by_field
    assert set(weights) == {field}


def test_base_uses_explicit_full_sequence_collator(runtime, tmp_path):
    train, validation = datasets(runtime)
    config = training.ExperimentConfig(dataset="uit_vsfc", output_root=str(tmp_path), protocol="revision", use_class_alpha=False, use_focal_loss=False, use_masking=False)
    trainer = training.make_trainer(Model(), Tokenizer(), train, validation, frame(), config)
    batch = trainer.kwargs["data_collator"]([Tokenizer()(train[0]["text"], add_special_tokens=False)])
    assert torch.equal(batch["labels"], batch["input_ids"])
    manifest = json.loads((__import__("pathlib").Path(config.output_dir) / "training_manifest.json").read_text())
    assert manifest["semantics"]["padding_mask"] == "attention_mask_equals_zero"
    assert manifest["semantics"]["true_eos_supervised"] is True
    assert "alpha_weights" not in batch


def test_alpha_without_mask_preserves_prompt_and_uses_response_only(runtime, tmp_path):
    train, validation = datasets(runtime)
    config = training.ExperimentConfig(dataset="uit_vsfc", output_root=str(tmp_path), protocol="revision", use_class_alpha=True, use_focal_loss=False, use_masking=False, alpha_scope="all_label_tokens")
    trainer = training.make_trainer(Model(), Tokenizer(), train, validation, frame(), config)
    feature = Tokenizer()(train[0]["text"], add_special_tokens=False)
    batch = trainer.kwargs["data_collator"]([feature])
    assert torch.equal(batch["labels"], batch["input_ids"])
    response_start = train[0]["text"].rfind("[/INST]") + len("[/INST]")
    assert torch.all(batch["alpha_weights"][0, :response_start] == 1)
    assert (batch["alpha_weights"][0, response_start:] != 1).any()
    with pytest.raises(ValueError, match="did not provide alpha_weights"):
        trainer.compute_loss(trainer.model, {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2), "labels": torch.tensor([[1, 2]])})
