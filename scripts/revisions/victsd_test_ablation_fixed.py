# %%
"""Test ViCTSD dùng chung cho các adapter; sửa ADAPTER_PATH rồi chạy main().

Chạy trong phiên inference có vLLM tương thích với PyTorch/CUDA đã cài.
Không chạy chung phiên còn giữ model/trainer 7B của bước huấn luyện.
Không loại hoặc tự cắt mẫu test. Mỗi lần chạy lưu một thư mục kết quả riêng.
"""

import os
import time
import json
import re
import math
import hashlib
import importlib.metadata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Đặt trước khi import vLLM; main guard ở cuối hỗ trợ bản .py dùng spawn.
os.environ.setdefault('VLLM_LOGGING_LEVEL', 'INFO')
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

# ==========================================
# 0. CẤU HÌNH — bạn đổi ADAPTER_PATH tại đây
# ==========================================
BASE_MODEL_PATH = 'Viet-Mistral/Vistral-7B-Chat'
ADAPTER_PATH = '/content/drive/MyDrive/label-focused-tuning/victsd/victsd_base_ce_r8_len1024_seed42/best_model'
TEST_CSV_PATH = '/content/drive/MyDrive/label-focused-tuning/victsd/ViCTSD_test.csv'

# None: lưu cạnh thư mục adapter, trong test_results/<adapter>_<thời điểm UTC>.
RESULTS_ROOT = None
MAX_NEW_TOKENS = 64
MAX_MODEL_LEN = 1024  # Tổng prompt + generation; giữ cùng giá trị cho mọi run.
INFERENCE_DTYPE = 'half'  # Giữ như code test gốc. Dùng cùng dtype cho mọi adapter.
GPU_MEMORY_UTILIZATION = 0.90
SEED = 42
WARMUP_SAMPLES = 4  # Warm-up được loại khỏi thời gian đo, không loại khỏi test.
SEPARATOR = ' [SEP] '
INVALID_LABEL = 'PARSE_ERROR'

SENTIMENT_MAP = {0: 'Non-toxic', 1: 'Toxic'}
SENTIMENT_LABELS = ['Non-toxic', 'Toxic']
TOPIC_LABELS = [
    'KinhDoanh', 'OtoXemay', 'TheGioi', 'TheThao', 'KhoaHoc',
    'ThoiSu', 'PhapLuat', 'GiaoDuc', 'SucKhoe', 'GiaiTri',
]
SYSTEM_PROMPT = (
    'Thực hiện đồng thời hai nhiệm vụ:\n'
    '(1) Sentiment: Non-toxic hoặc Toxic.\n'
    '(2) Topic: KinhDoanh, OtoXemay, TheGioi, TheThao, KhoaHoc, '
    'ThoiSu, PhapLuat, GiaoDuc, SucKhoe, GiaiTri.\n'
    'Chỉ trả về JSON với đúng hai khóa: Sentiment và Topic. Không giải thích thêm.'
)
FEW_SHOT_EXAMPLE = (
    'Ví dụ:\n'
    'Bình luận: "Biden dẫn Trump 13 điểm [SEP] Cuộc bầu cử ở Mỹ là công khai và công bằng nhất."\n'
    'Output: {"Sentiment": "Non-toxic", "Topic": "TheGioi"}'
)


# %%
# ==========================================
# 1. HELPER DỮ LIỆU VÀ TOKENIZER
# ==========================================
def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def prepare_test_frame(csv_path):
    import pandas as pd

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError('Test CSV rỗng.')
    missing = {'Toxicity', 'Topic'} - set(frame.columns)
    if missing:
        raise ValueError(f'Test CSV thiếu cột: {sorted(missing)}')
    frame = frame.copy()
    frame['source_row_index'] = list(range(len(frame)))
    toxicity = pd.to_numeric(frame['Toxicity'], errors='coerce')
    invalid = ~toxicity.isin(SENTIMENT_MAP)
    if invalid.any():
        raise ValueError(f'Toxicity phải là 0/1, lỗi ở dòng {frame.index[invalid].tolist()[:10]}')
    frame['Toxicity'] = toxicity.astype('int64')
    frame['Topic'] = frame['Topic'].fillna('').astype(str).str.strip()
    invalid = ~frame['Topic'].isin(TOPIC_LABELS)
    if invalid.any():
        raise ValueError(f'Topic không hợp lệ ở dòng {frame.index[invalid].tolist()[:10]}')

    # Dùng được cả CSV raw và CSV merged. Khi có cả hai, bắt buộc Text khớp train.
    if {'Title', 'Comment'}.issubset(frame.columns):
        title = frame['Title'].fillna('').astype(str).str.strip()
        comment = frame['Comment'].fillna('').astype(str).str.strip()
        merged = [f'{t}{SEPARATOR}{c}' if t else c for t, c in zip(title, comment)]
        if 'Text' in frame:
            existing = frame['Text'].fillna('').astype(str).str.strip().tolist()
            mismatches = [i for i, (left, right) in enumerate(zip(existing, merged)) if left.strip() != right.strip()]
            if mismatches:
                raise ValueError(
                    f'Text không khớp Title [SEP] Comment như train ở dòng {mismatches[:10]}. '
                    'Kiểm tra lại file merged; không tự thay đổi nội dung test.'
                )
        frame['Text'] = merged
        text_source = 'Title_Comment_validated_or_merged'
    elif 'Text' in frame:
        if not frame['Text'].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError('Cột Text có giá trị rỗng/không phải chuỗi.')
        # Giữ nguyên Text đã ghép; khoảng trắng cuối sau [SEP] có thể có từ train.
        text_source = 'premerged_Text_only_not_reconstructable_without_Title_Comment'
    else:
        raise ValueError('Test CSV cần Text hoặc cả Title và Comment.')
    if frame['Text'].eq('').any():
        raise ValueError('Có mẫu không có nội dung sau ghép Title/Comment.')
    return frame, text_source


def build_prompt(text, tokenizer):
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f'{FEW_SHOT_EXAMPLE}\n\nBình luận: "{text}"'},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def check_prompt_budget(token_ids, max_model_len, max_new_tokens):
    if max_new_tokens <= 0 or max_model_len <= max_new_tokens:
        raise ValueError('Cần 0 < MAX_NEW_TOKENS < MAX_MODEL_LEN.')
    if not token_ids or any(not ids for ids in token_ids):
        raise ValueError('Danh sách prompt rỗng hoặc có prompt không có token.')
    too_long = [
        {'row_index': i, 'prompt_tokens': len(ids), 'required_tokens': len(ids) + max_new_tokens}
        for i, ids in enumerate(token_ids) if len(ids) + max_new_tokens > max_model_len
    ]
    if too_long:
        raise ValueError(
            f'{len(too_long)} mẫu vượt MAX_MODEL_LEN={max_model_len} khi dành '
            f'{max_new_tokens} token cho câu trả lời. Ví dụ: {too_long[:10]}. '
            'Không loại/cắt mẫu. Nếu tăng MAX_MODEL_LEN, dùng cùng giá trị cho mọi adapter '
            'và giữ trong giới hạn context được backbone hỗ trợ.'
        )


def check_training_manifest(adapter_dir):
    manifest_path = adapter_dir.parent / 'run_manifest.json'
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    checks = {
        'model_id': BASE_MODEL_PATH,
        'system_prompt': SYSTEM_PROMPT,
    }
    for key, expected in checks.items():
        if key in manifest and manifest[key] != expected:
            raise ValueError(f'run_manifest.json: {key} không khớp cấu hình test.')
    for key in ('few_shot', 'few_shot_example'):
        if key in manifest and manifest[key] != FEW_SHOT_EXAMPLE:
            raise ValueError(f'run_manifest.json: {key} không khớp cấu hình test.')
    if 'dataset' in manifest and manifest['dataset'] != 'ViCTSD':
        raise ValueError('Adapter không thuộc ViCTSD theo run_manifest.json.')
    return {'path': str(manifest_path), 'sha256': file_sha256(manifest_path)}


# %%
# ==========================================
# 2. PARSER — phần này được kiểm tra độc lập, không cần GPU
# ==========================================
def parse_output(raw):
    """Parse one complete ViCTSD JSON response without coercing label values.

    Whitespace and one entire Markdown ```json ... ``` fence are accepted.
    Prose, multiple JSON documents, trailing text, duplicate keys, extra keys,
    and a non-object root invalidate both predictions. A missing or invalid
    individual field preserves the other valid field. Values must be strings
    in the corresponding allowed-label list after stripping whitespace.
    ``json_valid`` means one JSON object was fully decoded; ``schema_valid``
    additionally requires exactly the two expected keys and two valid labels.
    Uses SENTIMENT_LABELS, TOPIC_LABELS, and INVALID_LABEL from the caller.
    """
    result = {
        'pred_sentiment': INVALID_LABEL,
        'pred_topic': INVALID_LABEL,
        'json_valid': False,
        'sentiment_valid': False,
        'topic_valid': False,
        'schema_valid': False,
        'errors': [],
    }
    if not isinstance(raw, str):
        result['errors'].append('response_not_string')
        return result
    payload = raw.strip()
    if payload.startswith('```'):
        fenced = re.fullmatch(
            r'```json[ \t]*\r?\n(.*?)\r?\n```', payload,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced is None:
            result['errors'].append('invalid_json_fence')
            return result
        payload = fenced.group(1).strip()

    duplicate_keys = []

    def object_pairs(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                duplicate_keys.append(key)
            obj[key] = value
        return obj

    def reject_nonfinite(value):
        # Python accepts NaN/Infinity by default, but neither is valid JSON.
        raise ValueError('nonfinite_json_constant:' + value)

    try:
        parsed = json.loads(
            payload, object_pairs_hook=object_pairs,
            parse_constant=reject_nonfinite,
        )
    except (json.JSONDecodeError, ValueError, RecursionError):
        result['errors'].append('invalid_json')
        return result
    if not isinstance(parsed, dict):
        result['errors'].append('json_root_not_object')
        return result
    result['json_valid'] = True

    required = {'Sentiment', 'Topic'}
    extra_keys = set(parsed) - required
    if duplicate_keys:
        result['errors'].append('duplicate_keys:' + ','.join(sorted(set(duplicate_keys))))
    if extra_keys:
        result['errors'].append('extra_keys:' + ','.join(sorted(extra_keys)))
    # Duplicate/extra keys violate the complete response contract.
    if duplicate_keys or extra_keys:
        return result

    for field, labels, pred_key, valid_key in (
        ('Sentiment', SENTIMENT_LABELS, 'pred_sentiment', 'sentiment_valid'),
        ('Topic', TOPIC_LABELS, 'pred_topic', 'topic_valid'),
    ):
        if field not in parsed:
            result['errors'].append('missing_field:' + field)
            continue
        value = parsed[field]
        if not isinstance(value, str):
            result['errors'].append('field_not_string:' + field)
            continue
        value = value.strip()
        if value not in labels:
            result['errors'].append('invalid_label:' + field)
            continue
        result[pred_key] = value
        result[valid_key] = True

    result['schema_valid'] = (
        set(parsed) == required
        and result['sentiment_valid']
        and result['topic_valid']
    )
    return result


# %%
# ==========================================
# 3. METRICS — giữ tất cả mẫu, tập nhãn cố định
# ==========================================
def calculate_task_metrics(y_true, y_pred, labels):
    """F1 chuẩn theo TP/FP/FN; sentinel sai vẫn tạo FN cho lớp thật.

    Macro chỉ lấy mean trên các lớp nghiệp vụ cố định, không thêm sentinel
    thành một lớp thứ 3/11. Confusion matrix thêm cột sentinel để giữ đủ mẫu.
    """
    y_true, y_pred, labels = list(y_true), list(y_pred), list(labels)
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError('Hai mảng nhãn phải cùng độ dài và không rỗng.')
    if not labels or len(set(labels)) != len(labels) or INVALID_LABEL in labels:
        raise ValueError('Tập nhãn hợp lệ bị rỗng/trùng hoặc chứa sentinel.')
    if any(value not in labels for value in y_true):
        raise ValueError('Ground truth ngoài tập nhãn hợp lệ.')
    if any(value not in labels + [INVALID_LABEL] for value in y_pred):
        raise ValueError('Dự đoán cần được parser chuẩn hóa về nhãn hợp lệ hoặc sentinel.')

    per_class = {}
    counts = Counter(zip(y_true, y_pred))
    true_counts, pred_counts = Counter(y_true), Counter(y_pred)
    for label in labels:
        tp = counts[(label, label)]
        fp, fn = pred_counts[label] - tp, true_counts[label] - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class[label] = {
            'precision': precision, 'recall': recall, 'f1-score': f1,
            'support': int(true_counts[label]),
        }
    n = len(y_true)
    matrix_labels = labels + [INVALID_LABEL]
    matrix = [[int(counts[(actual, predicted)]) for predicted in matrix_labels] for actual in matrix_labels]
    assert sum(map(sum, matrix)) == n
    return {
        'n_samples': n,
        'invalid_predictions': int(pred_counts[INVALID_LABEL]),
        'valid_prediction_rate': 1.0 - pred_counts[INVALID_LABEL] / n,
        'accuracy': sum(left == right for left, right in zip(y_true, y_pred)) / n,
        'macro_f1': sum(per_class[label]['f1-score'] for label in labels) / len(labels),
        'weighted_f1': sum(per_class[label]['f1-score'] * true_counts[label] for label in labels) / n,
        'per_class': per_class,
        'confusion_matrix': {'labels': matrix_labels, 'counts': matrix},
    }


def verify_metrics_with_sklearn(y_true, y_pred, labels, metrics):
    """Đối chiếu scorer với sklearn trên chính toàn bộ kết quả, kể cả sentinel."""
    from sklearn.metrics import accuracy_score, f1_score

    reference = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
    }
    for key, expected in reference.items():
        if not math.isclose(metrics[key], float(expected), rel_tol=1e-10, abs_tol=1e-12):
            raise AssertionError(f'{key} không khớp sklearn: {metrics[key]} != {expected}')


def plot_confusion_matrices(sentiment_metrics, topic_metrics, output_dir):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, axes = plt.subplots(1, 2, figsize=(24, 9), dpi=150)
    for ax, name, metrics, cmap in (
        (axes[0], 'Sentiment', sentiment_metrics, 'Blues'),
        (axes[1], 'Topic', topic_metrics, 'Greens'),
    ):
        matrix = metrics['confusion_matrix']
        # Sentinel không là ground-truth; hàng cuối = 0. Cột cuối chứa output lỗi.
        ConfusionMatrixDisplay(
            np.asarray(matrix['counts']), display_labels=matrix['labels'],
        ).plot(ax=ax, colorbar=False, cmap=cmap, values_format='d')
        ax.set_title(f"{name} | All test | Acc={metrics['accuracy']:.3f} mF1={metrics['macro_f1']:.3f}")
        ax.tick_params(axis='x', rotation=50)
        pd.DataFrame(matrix['counts'], index=matrix['labels'], columns=matrix['labels']).to_csv(
            output_dir / f'confusion_matrix_{name.lower()}.csv', encoding='utf-8-sig',
        )
    fig.suptitle('ViCTSD — toàn bộ test; PARSE_ERROR = dự đoán không hợp lệ', fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / 'confusion_matrix_victsd.png', bbox_inches='tight')
    plt.show()
    plt.close(fig)


# %%
# ==========================================
# 4. KIỂM TRA NHỎ — không cần GPU hoặc sklearn
# ==========================================
def self_check():
    good = parse_output('{"Sentiment": "Toxic", "Topic": "TheGioi"}')
    assert good['schema_valid']
    partial = parse_output('{"Sentiment": "Toxic", "Topic": null}')
    assert partial['sentiment_valid'] and not partial['topic_valid']
    assert partial['pred_topic'] == INVALID_LABEL
    malformed = parse_output('{"Sentiment": "Toxic"')
    assert not malformed['json_valid']
    duplicate = parse_output('{"Sentiment":"Toxic","Sentiment":"Non-toxic","Topic":"TheGioi"}')
    assert not duplicate['schema_valid'] and duplicate['pred_sentiment'] == INVALID_LABEL
    unknown = parse_output('{"Sentiment":"Positive","Topic":"TheGioi"}')
    assert not unknown['sentiment_valid'] and unknown['topic_valid']

    # Hai dự đoán đúng + hai output hỏng: accuracy trên toàn test phải bằng 0.5.
    metrics = calculate_task_metrics(
        ['Non-toxic', 'Toxic', 'Non-toxic', 'Toxic'],
        ['Non-toxic', 'Toxic', INVALID_LABEL, INVALID_LABEL], SENTIMENT_LABELS,
    )
    assert metrics['n_samples'] == 4 and metrics['accuracy'] == 0.5
    assert math.isclose(metrics['macro_f1'], 2.0 / 3.0)
    assert metrics['invalid_predictions'] == 2
    assert sum(map(sum, metrics['confusion_matrix']['counts'])) == 4
    absent = calculate_task_metrics(['Non-toxic'], ['Non-toxic'], SENTIMENT_LABELS)
    assert absent['macro_f1'] == 0.5
    print('Kiểm tra parser, giữ đủ mẫu, fixed-label F1 và confusion matrix: OK.')


# %%
# ==========================================
# 5. INFERENCE VÀ LƯU KẾT QUẢ
# ==========================================
def main():
    self_check()
    # Mount khi chạy Colab, không phụ thuộc Path/import của một ô trước đó.
    if str(TEST_CSV_PATH).startswith('/content/drive/') and not Path('/content/drive/MyDrive').is_dir():
        try:
            from google.colab import drive
        except ImportError as error:
            raise FileNotFoundError('Đường dẫn Drive chỉ dùng được khi đã mount Drive trên Colab.') from error
        drive.mount('/content/drive')

    adapter_dir = Path(ADAPTER_PATH)
    adapter_config_path = adapter_dir / 'adapter_config.json'
    if not adapter_config_path.is_file():
        raise FileNotFoundError(f'Không thấy adapter_config.json tại {adapter_dir}; sửa ADAPTER_PATH.')
    adapter_weights = [path for name in ('adapter_model.safetensors', 'adapter_model.bin') if (path := adapter_dir / name).is_file()]
    if not adapter_weights:
        raise FileNotFoundError(f'Không thấy adapter_model.safetensors/.bin trong {adapter_dir}')
    if not Path(TEST_CSV_PATH).is_file():
        raise FileNotFoundError(TEST_CSV_PATH)
    if MAX_NEW_TOKENS <= 0 or MAX_MODEL_LEN <= MAX_NEW_TOKENS:
        raise ValueError('Cần 0 < MAX_NEW_TOKENS < MAX_MODEL_LEN.')
    if WARMUP_SAMPLES < 0:
        raise ValueError('WARMUP_SAMPLES phải không âm.')
    adapter_config = json.loads(adapter_config_path.read_text(encoding='utf-8'))
    if adapter_config.get('peft_type') != 'LORA':
        raise ValueError('Chỉ hỗ trợ adapter LoRA.')
    ranks = [int(adapter_config['r'])] + [int(value) for value in adapter_config.get('rank_pattern', {}).values()]
    required_rank = max(ranks)
    # Chọn mức hỗ trợ nhỏ nhất đủ chứa rank, thay vì luôn cấp phát rank=64.
    supported_ranks = (8, 16, 32, 64, 128, 256, 320, 512)
    max_lora_rank = next((rank for rank in supported_ranks if rank >= required_rank), None)
    if max_lora_rank is None or required_rank <= 0:
        raise ValueError(f'LoRA rank không được hỗ trợ bởi cấu hình test: {required_rank}')
    training_reference = check_training_manifest(adapter_dir)

    # Import thư viện nặng sau kiểm tra đường dẫn. Không tự cài hoặc đổi phiên bản.
    import pandas as pd
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if not torch.cuda.is_available():
        raise RuntimeError('Không có GPU CUDA.')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Cấu hình này dùng một GPU; giữ cùng cấu hình giữa các ablation.')
    print(f'PyTorch: {torch.__version__}; vLLM: {vllm.__version__}; CUDA: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM tổng: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB')
    print(f'Adapter: {adapter_dir}; rank thực={required_rank}; max_lora_rank={max_lora_rank}')

    tokenizer_source = str(adapter_dir) if (adapter_dir / 'tokenizer_config.json').is_file() else BASE_MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    frame, text_source = prepare_test_frame(TEST_CSV_PATH)
    texts = frame['Text'].tolist()
    prompts = [build_prompt(text, tokenizer) for text in texts]
    prompt_token_ids = []
    for start in range(0, len(prompts), 256):
        encoded = tokenizer(prompts[start:start + 256], add_special_tokens=False, truncation=False, padding=False)
        prompt_token_ids.extend(encoded['input_ids'])
    check_prompt_budget(prompt_token_ids, MAX_MODEL_LEN, MAX_NEW_TOKENS)
    inputs = [{'prompt_token_ids': ids} for ids in prompt_token_ids]
    lengths = [len(ids) for ids in prompt_token_ids]
    print(f'Test giữ đủ {len(frame)} mẫu; prompt min/max={min(lengths)}/{max(lengths)} token.')

    output_root = Path(RESULTS_ROOT) if RESULTS_ROOT else adapter_dir.parent / 'test_results'
    started_utc = datetime.now(timezone.utc)
    output_dir = output_root / f"{adapter_dir.name}_{started_utc.strftime('%Y%m%dT%H%M%S_%fZ')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    versions = {}
    for name in ('torch', 'vllm', 'transformers', 'numpy', 'pandas', 'scikit-learn', 'matplotlib'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'unknown'
    manifest = {
        'status': 'ready', 'started_utc': started_utc.isoformat(),
        'base_model': BASE_MODEL_PATH, 'adapter_path': str(adapter_dir),
        'adapter_config': adapter_config, 'adapter_config_sha256': file_sha256(adapter_config_path),
        'adapter_weight_sha256': {path.name: file_sha256(path) for path in adapter_weights},
        'training_manifest': training_reference,
        'test_csv_path': TEST_CSV_PATH, 'test_csv_sha256': file_sha256(TEST_CSV_PATH),
        'effective_test_sha256': hashlib.sha256(frame[['Text', 'Toxicity', 'Topic']].to_json(orient='records', force_ascii=False).encode('utf-8')).hexdigest(),
        'n_test_samples': len(frame), 'text_source': text_source,
        'tokenizer_source': tokenizer_source, 'add_special_tokens': False,
        'system_prompt': SYSTEM_PROMPT, 'few_shot': FEW_SHOT_EXAMPLE,
        'sentiment_labels': SENTIMENT_LABELS, 'topic_labels': TOPIC_LABELS,
        'dtype': INFERENCE_DTYPE, 'max_model_len': MAX_MODEL_LEN,
        'max_new_tokens': MAX_NEW_TOKENS, 'temperature': 0.0, 'seed': SEED,
        'stop': ['}'], 'include_stop_str_in_output': True,
        'max_lora_rank': max_lora_rank, 'gpu_memory_utilization': GPU_MEMORY_UTILIZATION,
        'enable_prefix_caching': False, 'enforce_eager': True,
        'warmup_samples': min(WARMUP_SAMPLES, len(inputs)),
        'length_policy': 'raise_before_model_load_if_any_prompt_plus_output_budget_exceeds_context; no_filter_no_truncation',
        'parser_policy': 'one_object_or_complete_json_fence; no_prose; no_extra_or_duplicate_keys; preserve_valid_field_when_other_missing_or_invalid',
        'metric_policy': 'all_test_rows; per_field_invalid_is_wrong; fixed_business_labels_for_macro_and_weighted_f1',
        'runtime_versions': versions,
    }
    save_json(output_dir / 'evaluation_manifest.json', manifest)
    print('Kết quả sẽ được lưu tại:', output_dir)

    print('Đang tải model với vLLM...')
    t_load = time.perf_counter()
    llm = LLM(
        model=BASE_MODEL_PATH, tokenizer=tokenizer_source,
        enable_lora=True, max_lora_rank=max_lora_rank,
        dtype=INFERENCE_DTYPE, gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN, enable_prefix_caching=False,
        enforce_eager=True, trust_remote_code=True, seed=SEED,
    )
    model_load_seconds = time.perf_counter() - t_load
    request = LoRARequest(lora_name='evaluated_adapter', lora_int_id=1, lora_path=str(adapter_dir))
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=MAX_NEW_TOKENS,
        stop=['}'], include_stop_str_in_output=True,
    )
    t_warmup = time.perf_counter()
    if WARMUP_SAMPLES:
        llm.generate(inputs[:min(WARMUP_SAMPLES, len(inputs))], sampling_params=sampling_params, lora_request=request, use_tqdm=False)
    warmup_seconds = time.perf_counter() - t_warmup

    # Hai snapshot toàn GPU: không đặt tên là peak, không coi là VRAM riêng của model.
    free_before, total_memory = torch.cuda.mem_get_info()
    t_inference = time.perf_counter()
    outputs = llm.generate(inputs, sampling_params=sampling_params, lora_request=request, use_tqdm=True)
    inference_seconds = time.perf_counter() - t_inference
    free_after, _ = torch.cuda.mem_get_info()
    if inference_seconds <= 0 or len(outputs) != len(frame):
        raise RuntimeError('Thời gian hoặc số lượng output không hợp lệ; không tính metric trên tập thiếu mẫu.')

    records = []
    raw_records = []
    for index, output in enumerate(outputs):
        if not output.outputs:
            raise RuntimeError(f'vLLM không trả completion cho dòng {index}.')
        completion = output.outputs[0]
        if output.prompt_token_ids is None or list(output.prompt_token_ids) != prompt_token_ids[index]:
            raise RuntimeError(f'Prompt token IDs/thứ tự output không khớp tại dòng {index}.')
        parsed = parse_output(completion.text)
        finish_reason = getattr(completion, 'finish_reason', None)
        raw_records.append({
            'source_row_index': index, 'request_id': str(output.request_id),
            'raw_model_output': completion.text,
            'prompt_token_ids': list(output.prompt_token_ids),
            'output_token_ids': list(completion.token_ids),
            'finish_reason': finish_reason,
            'stop_reason': getattr(completion, 'stop_reason', None),
        })
        records.append({
            'source_row_index': index, 'text': texts[index],
            'true_sentiment': SENTIMENT_MAP[int(frame.iloc[index]['Toxicity'])],
            'pred_sentiment': parsed['pred_sentiment'],
            'true_topic': frame.iloc[index]['Topic'], 'pred_topic': parsed['pred_topic'],
            'json_valid': parsed['json_valid'], 'schema_valid': parsed['schema_valid'],
            'sentiment_valid': parsed['sentiment_valid'], 'topic_valid': parsed['topic_valid'],
            'parse_errors': ';'.join(parsed['errors']),
            'prompt_tokens': len(prompt_token_ids[index]), 'output_tokens': len(completion.token_ids),
            'finish_reason': finish_reason, 'stopped_by_length_limit': finish_reason == 'length',
            'stop_reason': getattr(completion, 'stop_reason', None),
            'raw_model_output': completion.text,
        })

    # Lưu raw trước report/vẽ hình để vẫn giữ dự đoán nếu bước báo cáo gặp lỗi.
    results = pd.DataFrame(records)
    results.to_csv(output_dir / 'test_predictions.csv', index=False, encoding='utf-8-sig')
    with open(output_dir / 'raw_outputs.jsonl', 'w', encoding='utf-8') as handle:
        for record in raw_records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')

    true_s, pred_s = results['true_sentiment'].tolist(), results['pred_sentiment'].tolist()
    true_t, pred_t = results['true_topic'].tolist(), results['pred_topic'].tolist()
    sentiment_metrics = calculate_task_metrics(true_s, pred_s, SENTIMENT_LABELS)
    topic_metrics = calculate_task_metrics(true_t, pred_t, TOPIC_LABELS)
    verify_metrics_with_sklearn(true_s, pred_s, SENTIMENT_LABELS, sentiment_metrics)
    verify_metrics_with_sklearn(true_t, pred_t, TOPIC_LABELS, topic_metrics)
    n = len(results)
    diagnostics = {
        'n_test_samples': n,
        'invalid_json_object_count': sum(not row['json_valid'] for row in records),
        'invalid_schema_or_label_count': sum(not row['schema_valid'] for row in records),
        'sentiment_invalid_count': sum(not row['sentiment_valid'] for row in records),
        'topic_invalid_count': sum(not row['topic_valid'] for row in records),
        'both_fields_valid_count': sum(row['sentiment_valid'] and row['topic_valid'] for row in records),
        'schema_valid_rate': sum(row['schema_valid'] for row in records) / n,
        'finish_reason_counts': dict(Counter(str(row['finish_reason']) for row in records)),
        'length_limited_count': sum(row['stopped_by_length_limit'] for row in records),
    }
    all_metrics = {
        'scope': 'entire_test_set_no_rows_removed',
        'sentiment': sentiment_metrics, 'topic': topic_metrics,
        'joint_exact_match_accuracy': sum(
            left_s == right_s and left_t == right_t
            for left_s, right_s, left_t, right_t in zip(true_s, pred_s, true_t, pred_t)
        ) / n,
        'diagnostics': diagnostics,
    }
    total_input = int(results['prompt_tokens'].sum())
    total_output = int(results['output_tokens'].sum())
    performance = {
        'hardware': {
            'gpu_name': torch.cuda.get_device_name(0),
            'total_vram_gib': total_memory / 1024**3,
            'device_memory_used_before_inference_gib': (total_memory - free_before) / 1024**3,
            'device_memory_used_after_inference_gib': (total_memory - free_after) / 1024**3,
            'memory_measurement_note': 'global_device_snapshots_include_vLLM_KV_cache_and_other_processes; not_peak_or_model_only',
        },
        'timing': {
            'model_load_wall_seconds': model_load_seconds,
            'warmup_wall_seconds': warmup_seconds,
            'inference_wall_seconds_excluding_warmup': inference_seconds,
            'amortized_seconds_per_sample': inference_seconds / n,
            'samples_per_second': n / inference_seconds,
            'timing_note': 'whole_synchronous_generate_call; amortized_time_is_not_individual_request_latency',
        },
        'aggregate_token_throughput': {
            'total_input_tokens': total_input, 'total_output_tokens': total_output,
            'input_tokens_per_total_inference_second': total_input / inference_seconds,
            'output_tokens_per_total_inference_second': total_output / inference_seconds,
            'all_tokens_per_total_inference_second': (total_input + total_output) / inference_seconds,
            'note': 'denominator_is_total_inference_wall_time; not_separate_prefill_or_decode_speed',
        },
    }
    save_json(output_dir / 'metrics.json', all_metrics)
    save_json(output_dir / 'performance_metrics.json', performance)
    manifest.update(status='scored', inference_wall_seconds=inference_seconds, outputs_count=len(outputs))
    save_json(output_dir / 'evaluation_manifest.json', manifest)

    from sklearn.metrics import classification_report
    for name, actual, predicted, labels, metrics in (
        ('Sentiment', true_s, pred_s, SENTIMENT_LABELS, sentiment_metrics),
        ('Topic', true_t, pred_t, TOPIC_LABELS, topic_metrics),
    ):
        print(f"\n{name} — toàn bộ {n} mẫu, output không hợp lệ={metrics['invalid_predictions']}")
        print(f"Accuracy={metrics['accuracy']:.4f}; Macro F1={metrics['macro_f1']:.4f}; Weighted F1={metrics['weighted_f1']:.4f}")
        report = classification_report(actual, predicted, labels=labels, zero_division=0, digits=4)
        print(report)
        (output_dir / f'classification_report_{name.lower()}.txt').write_text(report, encoding='utf-8')
    print(f"Joint exact-match accuracy: {all_metrics['joint_exact_match_accuracy']:.4f}")
    print('Diagnostics:', json.dumps(diagnostics, ensure_ascii=False))
    print(f'Inference: {inference_seconds:.2f} s; {n / inference_seconds:.2f} mẫu/s')
    print(f'Thời gian bình quân theo tổng lượt chạy: {inference_seconds / n:.4f} s/mẫu (không phải request latency).')
    plot_confusion_matrices(sentiment_metrics, topic_metrics, output_dir)
    manifest['status'] = 'completed'
    save_json(output_dir / 'evaluation_manifest.json', manifest)
    print('\nĐã lưu predictions, raw outputs, metrics, performance, confusion matrices và manifest tại:')
    print(output_dir)
    return str(output_dir)


# %%
# Chạy ô này sau khi đã sửa ADAPTER_PATH ở phần cấu hình.
if __name__ == '__main__':
    main()
