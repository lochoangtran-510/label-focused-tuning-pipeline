# DÁN CẢ Ô NÀY VÀO CUỐI NOTEBOOK HUẤN LUYỆN NEU.
# Chỉ thay 3 đường dẫn dưới đây bằng checkpoint/adapter bạn muốn đánh giá.
CHECKPOINT_DIRS = [
    "DAN_DUONG_DAN_CHECKPOINT_1_VAO_DAY",
    "DAN_DUONG_DAN_CHECKPOINT_2_VAO_DAY",
    "DAN_DUONG_DAN_CHECKPOINT_3_VAO_DAY",
]
BATCH_SIZE_VAL = 4
MAX_NEW_TOKENS_VAL = 64

import gc, json, time, uuid
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import torch
from tqdm.auto import tqdm
from peft import PeftModel
from sklearn.metrics import f1_score

# Dùng đúng tokenizer, prompt, nhãn và đường dẫn đã có trong notebook huấn luyện.
_required = ["tokenizer", "SYSTEM_PROMPT", "FEW_SHOT_EXAMPLE", "sentiment_map",
             "topic_map", "SENTIMENT_LABELS", "TOPIC_LABELS", "TRAIN_CSV_PATH", "VAL_CSV_PATH"]
_missing = [name for name in _required if name not in globals()]
if _missing:
    raise RuntimeError(f"Thiếu {_missing}. Chạy ô này trong cùng phiên Python sau huấn luyện NEU.")
if len(CHECKPOINT_DIRS) != 3 or len(set(CHECKPOINT_DIRS)) != 3:
    raise ValueError("Cần đúng 3 đường dẫn checkpoint khác nhau.")
if BATCH_SIZE_VAL < 1:
    raise ValueError("BATCH_SIZE_VAL phải >= 1.")

_candidates = [getattr(globals().get("trainer"), "model", None),
               globals().get("model_with_lora"), globals().get("model")]
_val_model = next((m for m in _candidates if isinstance(m, PeftModel)), None)
if _val_model is None:
    raise RuntimeError("Không còn model LoRA trong RAM. Ô này dùng lại model sau train, không tải thêm model 7B.")
_original_adapter = _val_model.active_adapter
if not isinstance(_original_adapter, str):
    raise ValueError("Cần một adapter gốc đang active để khôi phục model sau đánh giá.")
_base_id = _val_model.peft_config[_original_adapter].base_model_name_or_path
for _path in CHECKPOINT_DIRS:
    _dir = Path(_path)
    if not (_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Không tìm thấy adapter_config.json: {_dir}")
    if not any((_dir / name).is_file() for name in ["adapter_model.safetensors", "adapter_model.bin"]):
        raise FileNotFoundError(f"Không tìm thấy trọng số LoRA: {_dir}")
    _cfg = json.loads((_dir / "adapter_config.json").read_text())
    if _cfg.get("peft_type") != "LORA" or _cfg.get("task_type") != "CAUSAL_LM":
        raise ValueError(f"Checkpoint không phải LoRA CAUSAL_LM: {_dir}")
    if _cfg.get("base_model_name_or_path") != _base_id:
        raise ValueError(f"Checkpoint dùng base model khác model đang có trong RAM: {_dir}")

def _map_gold(values, mapping, labels, field):
    result = [mapping.get(str(value).strip(), str(value).strip()) for value in values]
    bad = sorted(set(result) - set(labels))
    if bad:
        raise ValueError(f"Nhãn thật {field} không hợp lệ: {bad}; không tự thay nhãn thật.")
    return result

def _build_val_prompt(text):
    # Không dùng generate_train_prompt: hàm đó có assistant chứa nhãn thật.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'{FEW_SHOT_EXAMPLE}\n\nPhản hồi: "{text}"'},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def _parse_val_output(raw):
    try:
        start, end = raw.rfind("{"), raw.rfind("}") + 1
        obj = json.loads(raw[start:end]) if start >= 0 and end > start else {}
        if not isinstance(obj, dict):
            return "", ""
        return str(obj.get("Sentiment", "")).strip(), str(obj.get("Topic", "")).strip()
    except (ValueError, TypeError):
        return "", ""

_val_df = pd.read_csv(VAL_CSV_PATH)
_train_df_for_fallback = pd.read_csv(TRAIN_CSV_PATH)
for _name, _df in [("validation", _val_df), ("train", _train_df_for_fallback)]:
    _needed = {"sentiment", "classification"} | ({"text"} if _name == "validation" else set())
    if _df.empty or not _needed.issubset(_df.columns):
        raise ValueError(f"CSV {_name} rỗng hoặc thiếu cột {_needed - set(_df.columns)}.")
if _val_df["text"].isna().any():
    raise ValueError("Validation có text thiếu; dừng để không bỏ mẫu hoặc sinh từ dữ liệu thiếu.")

_true_s = _map_gold(_val_df["sentiment"], sentiment_map, SENTIMENT_LABELS, "Sentiment")
_true_t = _map_gold(_val_df["classification"], topic_map, TOPIC_LABELS, "Topic")
_train_s = _map_gold(_train_df_for_fallback["sentiment"], sentiment_map, SENTIMENT_LABELS, "train Sentiment")
_train_t = _map_gold(_train_df_for_fallback["classification"], topic_map, TOPIC_LABELS, "train Topic")
_majority_s = pd.Series(_train_s).mode().iloc[0]
_majority_t = pd.Series(_train_t).mode().iloc[0]
_prompts = [_build_val_prompt(text) for text in _val_df["text"].astype(str)]
_lengths = [len(ids) for ids in tokenizer(_prompts, add_special_tokens=False, truncation=False)["input_ids"]]
_context_limit = getattr(_val_model.config, "max_position_embeddings", None)
if _context_limit and max(_lengths) + MAX_NEW_TOKENS_VAL > _context_limit:
    raise ValueError("Có prompt vượt context của model; ô này không cắt hay bỏ mẫu validation.")

_save_dir = Path(CHECKPOINT_DIRS[0]).parent / ("validation_mf1_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
_save_dir.mkdir(parents=True, exist_ok=False)
_old_padding = tokenizer.padding_side
_old_pad_token = tokenizer.pad_token
_old_training = _val_model.training
_old_cache = _val_model.config.use_cache
_old_grad_flags = {name: param.requires_grad for name, param in _val_model.named_parameters()}
_device = _val_model.get_input_embeddings().weight.device
_val_results = []
print(f"Validation: {len(_val_df)} mẫu | fallback từ train: {_majority_s} / {_majority_t}")
print("Chọn checkpoint theo (Sentiment Macro F1 + Topic Macro F1) / 2.")

try:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    _val_model.eval()
    _val_model.config.use_cache = True
    for _index, _checkpoint in enumerate(CHECKPOINT_DIRS, 1):
        _adapter_name = f"val_mf1_{_index}_{uuid.uuid4().hex[:8]}"
        _raw_outputs = []
        _start_time = time.perf_counter()
        try:
            _val_model.load_adapter(_checkpoint, adapter_name=_adapter_name, is_trainable=False)
            _val_model.set_adapter(_adapter_name)
            _val_model.eval()
            for _start in tqdm(range(0, len(_prompts), BATCH_SIZE_VAL), desc=f"VAL checkpoint {_index}/3"):
                _batch_prompts = _prompts[_start:_start + BATCH_SIZE_VAL]
                _encoded = tokenizer(_batch_prompts, return_tensors="pt", padding=True,
                                     truncation=False, add_special_tokens=False)
                _encoded = {key: value.to(_device) for key, value in _encoded.items()
                            if key in {"input_ids", "attention_mask"}}
                _prompt_width = _encoded["input_ids"].shape[1]
                with torch.inference_mode():
                    _generated = _val_model.generate(
                        **_encoded, max_new_tokens=MAX_NEW_TOKENS_VAL,
                        do_sample=False, num_beams=1, num_return_sequences=1,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                        use_cache=True, return_dict_in_generate=False,
                    )
                _raw_outputs.extend(tokenizer.batch_decode(
                    _generated[:, _prompt_width:].cpu(), skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ))
                del _encoded, _generated
        finally:
            _val_model.set_adapter(_original_adapter)
            if _adapter_name in _val_model.peft_config:
                _val_model.base_model.delete_adapter(_adapter_name)
                _val_model.set_adapter(_original_adapter)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(_raw_outputs) != len(_val_df):
            raise RuntimeError("Số dự đoán không khớp toàn bộ validation.")
        _parsed = [_parse_val_output(raw) for raw in _raw_outputs]
        _invalid_s = [sent not in SENTIMENT_LABELS for sent, topic in _parsed]
        _invalid_t = [topic not in TOPIC_LABELS for sent, topic in _parsed]
        _pred_s = [sent if not bad else _majority_s for (sent, topic), bad in zip(_parsed, _invalid_s)]
        _pred_t = [topic if not bad else _majority_t for (sent, topic), bad in zip(_parsed, _invalid_t)]
        _mf1_s = float(f1_score(_true_s, _pred_s, labels=SENTIMENT_LABELS, average="macro", zero_division=0))
        _mf1_t = float(f1_score(_true_t, _pred_t, labels=TOPIC_LABELS, average="macro", zero_division=0))
        _val_results.append({
            "checkpoint": _checkpoint, "n_val": len(_val_df),
            "sentiment_mf1": _mf1_s, "topic_mf1": _mf1_t,
            "mean_mf1": (_mf1_s + _mf1_t) / 2,
            "invalid_sentiment": sum(_invalid_s), "invalid_topic": sum(_invalid_t),
            "invalid_any": sum(a or b for a, b in zip(_invalid_s, _invalid_t)),
            "seconds": time.perf_counter() - _start_time,
        })
        pd.DataFrame({
            "row_index": range(len(_val_df)), "text": _val_df["text"].tolist(),
            "raw_output": _raw_outputs, "true_sentiment": _true_s, "true_topic": _true_t,
            "pred_sentiment_original": [sent for sent, topic in _parsed],
            "pred_topic_original": [topic for sent, topic in _parsed],
            "pred_sentiment": _pred_s, "pred_topic": _pred_t,
            "sentiment_invalid": _invalid_s, "topic_invalid": _invalid_t,
        }).to_csv(_save_dir / f"checkpoint_{_index}_val_predictions.csv", index=False, encoding="utf-8-sig")
        # Lưu ngay từng checkpoint để kết quả hoàn tất không mất nếu checkpoint sau gặp lỗi.
        pd.DataFrame(_val_results).to_csv(_save_dir / "val_results.csv", index=False, encoding="utf-8-sig")
        print(f"  Sentiment={_mf1_s:.4f} | Topic={_mf1_t:.4f} | Mean={(_mf1_s + _mf1_t)/2:.4f}")
finally:
    _val_model.set_adapter(_original_adapter)
    for _name, _param in _val_model.named_parameters():
        if _name in _old_grad_flags:
            _param.requires_grad_(_old_grad_flags[_name])
    _val_model.train(_old_training)
    _val_model.config.use_cache = _old_cache
    tokenizer.padding_side = _old_padding
    tokenizer.pad_token = _old_pad_token

# Xếp hạng bằng giá trị chưa làm tròn; nếu hòa, giữ thứ tự 3 đường dẫn bạn đã nhập.
VAL_CHECKPOINT_RESULTS = pd.DataFrame(_val_results).sort_values("mean_mf1", ascending=False, kind="mergesort").reset_index(drop=True)
BEST_VAL_CHECKPOINT = VAL_CHECKPOINT_RESULTS.loc[0, "checkpoint"]
VAL_CHECKPOINT_RESULTS.to_csv(_save_dir / "val_ranking.csv", index=False, encoding="utf-8-sig")
with open(_save_dir / "selection.json", "w", encoding="utf-8") as _file:
    json.dump({"best_checkpoint": BEST_VAL_CHECKPOINT,
               "criterion": "mean_of_sentiment_and_topic_validation_macro_f1",
               "validation_csv": str(VAL_CSV_PATH), "train_csv_for_fallback": str(TRAIN_CSV_PATH),
               "majority_sentiment": _majority_s, "majority_topic": _majority_t,
               "max_new_tokens": MAX_NEW_TOKENS_VAL, "n_val": len(_val_df),
               "results": VAL_CHECKPOINT_RESULTS.to_dict(orient="records")},
              _file, ensure_ascii=False, indent=2)
print("\n=== KẾT QUẢ VALIDATION — Macro F1 dạng 0–1 ===")
print(VAL_CHECKPOINT_RESULTS.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\n✅ Checkpoint tốt nhất theo Mean Macro F1: {BEST_VAL_CHECKPOINT}")
print(f"💾 Toàn bộ dự đoán và bảng xếp hạng: {_save_dir}")
