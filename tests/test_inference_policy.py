"""Inference keeps invalid predictions without consulting training labels."""
import json
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from label_focused.datasets import DATASET_REGISTRY
from label_focused import inference


def make_test_frame(dataset):
    spec = DATASET_REGISTRY[dataset]
    return pd.DataFrame({
        spec.text_column: ["first test row", "second test row"],
        spec.sentiment_column: [0, 1],
        spec.topic_column: list(spec.topic_labels[:2]) if dataset == "victsd" else [0, 1],
    }, index=[17, 8])


def mock_generation(monkeypatch, dataset, architecture, partial=False):
    spec = DATASET_REGISTRY[dataset]
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: json.dumps(messages, ensure_ascii=False))
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    vllm = ModuleType("vllm")
    class LLM:
        def __init__(self, **kwargs): self.calls = 0
        def generate(self, prompts, sampling, **kwargs):
            self.calls += 1
            if architecture == "dual_adapter":
                field, value = ("Sentiment", "outside_label_set") if self.calls == 1 else ("Topic", spec.topic_labels[0].lower())
            else:
                field = spec.zero_shot_sentiment_key if architecture == "zero_shot" else "Sentiment"
                value = "outside_label_set"
            first = {field: value}
            if architecture != "dual_adapter": first["Topic"] = spec.topic_labels[0].lower()
            texts = [json.dumps(first), "not JSON"]
            if partial: texts = texts[:1]
            return [SimpleNamespace(prompt_token_ids=[1, 2], outputs=[SimpleNamespace(text=text, token_ids=[3])]) for text in texts]
    vllm.LLM = LLM
    vllm.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    lora = ModuleType("vllm.lora"); request = ModuleType("vllm.lora.request")
    request.LoRARequest = lambda *args: args
    monkeypatch.setitem(sys.modules, "vllm.lora", lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", request)
    class Monitor:
        peak_gb = None
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(inference, "VramMonitor", Monitor)


@pytest.mark.parametrize("dataset", ["neu_esc", "uit_vsfc", "victsd"])
@pytest.mark.parametrize("architecture", ["zero_shot", "joint", "dual_adapter"])
def test_all_datasets_and_architectures_disable_fallback(monkeypatch, dataset, architecture):
    mock_generation(monkeypatch, dataset, architecture)
    config = inference.InferenceConfig(dataset=dataset, architecture=architecture, joint_adapter="joint", sentiment_adapter="sentiment", topic_adapter="topic")
    frame = make_test_frame(dataset)
    class ForbiddenTrainingData:
        def iterrows(self): raise AssertionError("Inference must not consult training labels")
    predictions, performance = inference.run_inference(frame, ForbiddenTrainingData(), config)
    # Empty and differently labelled training inputs cannot alter predictions.
    other, _ = inference.run_inference(frame, pd.DataFrame(), config)
    preferred, _ = inference.run_inference(frame, config=config)
    pd.testing.assert_frame_equal(predictions, other)
    pd.testing.assert_frame_equal(predictions, preferred)
    assert len(predictions) == len(frame) == 2
    assert predictions["text"].tolist() == frame[DATASET_REGISTRY[dataset].text_column].tolist()
    assert predictions["pred_sentiment"].tolist() == ["PARSE_ERROR", "PARSE_ERROR"]
    assert predictions["pred_topic"].tolist() == [DATASET_REGISTRY[dataset].topic_labels[0], "PARSE_ERROR"]
    assert predictions["pred_sentiment_original"].tolist() == ["outside_label_set", "PARSE_ERROR"]
    assert predictions["parse_status"].tolist() == ["SENTIMENT_INVALID", "INVALID_JSON"]
    assert performance["invalid_counts"] == {"sentiment": 2, "topic": 1, "rows_with_any_invalid": 2}
    assert "majority_fallback" not in performance and "fallback_policy" not in performance
    assert performance["evaluation_policy"]["fallback_enabled"] is False
    assert performance["evaluation_policy"]["fallback"] == "disabled"
    assert performance["evaluation_policy"]["invalid_sentinel"] == "PARSE_ERROR"


def test_partial_generation_is_rejected_without_dropping_rows(monkeypatch):
    mock_generation(monkeypatch, "uit_vsfc", "joint", partial=True)
    config = inference.InferenceConfig(dataset="uit_vsfc", architecture="joint", joint_adapter="joint")
    with pytest.raises(ValueError, match="every test row"):
        inference.run_inference(make_test_frame("uit_vsfc"), config=config)
