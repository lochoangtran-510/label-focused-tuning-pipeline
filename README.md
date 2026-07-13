# Label-Focused Tuning Pipeline

Official source code for *Label-Focused Tuning Pipeline for Efficient Multitask
Text Classification in a Low-Resource Language*.

The repository provides shared Joint LoRA and Dual Adapter training pipelines for three
Vietnamese multitask datasets:

- NEU-ESC: sentiment and educational-topic classification.
- UIT-VSFC: sentiment and feedback-topic classification.
- ViCTSD: toxicity and news-topic classification.

Dataset-specific behavior is isolated in
`src/label_focused/datasets.py`; model loading, LoRA, completion-only masking,
and alpha-balanced focal loss are shared by all experiments.

## Repository structure

```text
.
├── configs/{datasets,experiments,rank_sweeps}/
├── data/{raw,processed}/                   # downloaded cache; not redistributed
├── notebooks/colab_quickstart.ipynb       # Colab quickstart
├── scripts/{prepare_data,train,predict,evaluate}.py
├── checkpoints/README.md                   # LoRA release layout
├── results/                                # predictions, metrics, figures
├── src/label_focused/
│   ├── datasets.py                        # schemas, labels, loaders
│   ├── prompting.py                       # Joint/Dual prompt and JSON parser
│   ├── completion_collator.py             # response-only masking
│   ├── losses.py                          # alpha-balanced focal loss
│   ├── training.py                        # Joint/Dual LoRA training
│   ├── inference.py                       # zero-shot/Joint/Dual vLLM
│   └── evaluation.py                      # full-test metrics and fallback
├── EXPERIMENTS.md                         # exact reported experiment scope
├── requirements.txt
└── CITATION.cff
```

Exploratory Colab notebooks and the manuscript PDF are excluded from the public
repository. The clean notebook above is the only notebook entry point.
See [`EXPERIMENTS.md`](EXPERIMENTS.md) before launching a reproduction run.

## Installation

Python 3.10+ and a CUDA GPU with bfloat16 support are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Training and vLLM inference use separate environments because their pinned
`transformers` versions differ. Create a second environment for inference:

```bash
python -m venv .venv-inference
source .venv-inference/bin/activate
pip install -r requirements-inference.txt
```

Prepare the datasets as described in [`data/README.md`](data/README.md):

```bash
python scripts/prepare_data.py --dataset neu_esc
python scripts/prepare_data.py --dataset uit_vsfc
python scripts/prepare_data.py --dataset victsd
```

Then
open `notebooks/colab_quickstart.ipynb`. Select a dataset in the
configuration cell:

```python
DATASET_NAME = "neu_esc"  # "neu_esc", "uit_vsfc", or "victsd"
```

All three datasets are downloaded from Hugging Face while preserving their
official train/validation/test splits. NEU-ESC is gated: accept its access
conditions at <https://huggingface.co/datasets/hung20gg/NEU-ESC>, then run
`hf auth login` (or provide `HF_TOKEN` through a private Colab secret).
No Hugging Face token is stored in the source.

## YAML-driven training

All public reproduction presets train for **3 complete epochs**. They set
`max_steps: -1`, so `TrainingArguments` does not replace the epoch schedule with
a fixed step limit.

Run a reported configuration with:

```bash
python scripts/train.py \
  --config configs/ablation/neu_esc_full_r32.yaml
```

The same run can be composed from readable dataset and method configs:

```bash
python scripts/train.py \
  --dataset-config configs/datasets/neu_esc.yaml \
  --experiment-config configs/experiments/joint_pipeline.yaml
```

For a quick pipeline check only, explicitly override the schedule:

```bash
python scripts/train.py \
  --config configs/ablation/neu_esc_full_r32.yaml \
  --max-steps 2 \
  --output-root outputs/smoke
```

The override is intended only for smoke testing and must not be used for
reported results. Every run records `resolved_config.json`, `environment.json`,
`training_metrics.json`, `trainer_log_history.json`, and `run_summary.json`.

## Reproducing configurations

The full Joint method uses:

```python
ExperimentConfig(
    dataset="neu_esc",
    lora_rank=32,
    use_masking=True,
    use_focal_loss=True,
)
```

For two fully independent task adapters, use:

```python
ExperimentConfig(
    dataset="neu_esc",
    architecture="dual_adapter",
    sentiment_lora_rank=32,
    topic_lora_rank=32,
    use_masking=True,
    use_focal_loss=True,
)
```

The Sentiment model and Topic model are initialized from the same frozen base
model but trained and saved independently. Different task ranks are supported
for asymmetric Dual Adapter experiments.

## Inference and full-test evaluation

Joint inference performs one generation per sample:

```bash
python scripts/predict.py \
  --dataset neu_esc \
  --architecture joint \
  --joint-adapter outputs/neu_esc/joint/full_r32/best_model \
  --output-dir outputs/neu_esc/joint/full_r32/inference
```

Zero-shot uses the same two-task output contract, but omits both the one-shot
example and the LoRA adapter:

```bash
python scripts/predict.py \
  --dataset neu_esc \
  --architecture zero_shot \
  --output-dir outputs/neu_esc/zero_shot
```

Dual Adapter inference runs the independent Sentiment and Topic adapters on the
same ordered test split:

```bash
python scripts/predict.py \
  --dataset neu_esc \
  --architecture dual_adapter \
  --sentiment-adapter PATH_TO_SENTIMENT_ADAPTER \
  --topic-adapter PATH_TO_TOPIC_ADAPTER \
  --output-dir outputs/neu_esc/dual_adapter/inference
```

Evaluate a saved prediction file with the same full-test fallback policy:

```bash
python scripts/evaluate.py \
  --dataset neu_esc \
  --predictions outputs/neu_esc/dual_adapter/inference/predictions.csv
```

Malformed JSON, empty outputs, and labels outside the dataset label set are
replaced **independently per task** by the majority class calculated from the
training split. No test row is discarded. `predictions.csv` retains raw output,
original parsed labels, invalid flags, and corrected labels for auditing.

Inference writes:

- `predictions.csv`: every test sample and corrected predictions.
- `invalid_predictions.csv`: audit subset only; it is not the evaluation set.
- `performance.json`: elapsed time, input/output throughput, samples/second,
  parse-invalid counts, majority fallbacks, and peak GPU memory sampled by NVML.

Generate confusion matrices and the task-error distribution without rerunning
the model:

```bash
python scripts/generate_figures.py \
  --dataset neu_esc \
  --predictions results/predictions/neu_esc/joint_pipeline_r32.csv \
  --prefix neu_joint_pipeline_r32
```

The mapping from manuscript tables/figures to artifacts is documented in
[`results/metrics/paper_results_mapping.md`](results/metrics/paper_results_mapping.md).
Verified LoRA release requirements are documented in
[`checkpoints/README.md`](checkpoints/README.md). Checkpoints and numerical CSVs
are intentionally not fabricated; they must be added from audited experiment runs.

Run the evaluation-policy tests in the training/development environment:

```bash
pip install -r requirements-dev.txt
pytest
```

The implementation is reusable across datasets, but the paper did **not** evaluate
every possible dataset/configuration combination. Use the experiment matrix below
when reproducing reported results.

| Analysis | NEU-ESC | UIT-VSFC | ViCTSD |
|---|---|---|---|
| Main Joint/Dual comparison | Yes | Yes | Yes |
| Joint ablation (Base, Masking, Full) | r=32 | r=8 | r=8 |
| Joint rank sensitivity (8, 16, 32, 64) | Yes | Yes | No |
| Dual Adapter rank sensitivity | Yes | No | No |
| Inference efficiency | No | Yes | No |
| Detailed error analysis | Yes | No | No |

The Dual Adapter rank analysis on NEU-ESC used these Sentiment/Topic pairs:
`8/8`, `16/16`, `32/32`, `64/64`, `16/64`, `64/16`, and `32/8`.
The NEU-ESC Dual + Pipeline row in the main comparison uses `r=16/64`.
The ablation reported only Base, Masking, and Full Pipeline; a standalone
Focal-only configuration was not reported as a separate row. Outputs are written
under `outputs/<dataset>/<architecture>/<configuration>/` and excluded from Git.

## Reproducing tables and figures

Run the exact reported ablation or rank matrix with:

```bash
python scripts/run_ablation.py --dataset neu_esc
python scripts/run_rank_sweep.py --architecture joint --dataset neu_esc
python scripts/run_rank_sweep.py --architecture dual_adapter --dataset neu_esc
```

Use `--dry-run` to inspect the resolved commands without launching training.
Training is followed by `scripts/predict.py` and `scripts/evaluate.py`; table and
figure provenance is listed in
[`paper_results_mapping.md`](results/metrics/paper_results_mapping.md).

## Pretrained LoRA adapters

The base Vistral model is downloaded from Hugging Face and is not redistributed.
The six principal Joint/Dual LoRA adapters will be attached to the audited
Zenodo release. See [`checkpoints/README.md`](checkpoints/README.md) for the
expected layout. Until those files are added, the repository provides training
and inference code but does not claim that placeholder checkpoints reproduce
the manuscript numbers.

## Data availability

The datasets are not redistributed in this Git repository. The preparation
script downloads their Hugging Face copies and preserves the published splits:

- UIT-VSFC: <https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback>
- NEU-ESC (gated): <https://huggingface.co/datasets/hung20gg/NEU-ESC>
- ViCTSD: <https://huggingface.co/datasets/tarudesu/ViCTSD>

Please also cite the original publications:

- UIT-VSFC: <https://doi.org/10.1109/KSE.2018.8573337>
- NEU-ESC: <https://doi.org/10.48550/arXiv.2506.23524>
- ViCTSD: <https://doi.org/10.1007/978-3-030-79457-6_49>

## Code availability

Our source code is available on GitHub and Zenodo as follows:

- GitHub: <https://github.com/lochoangtran-510/label-focused-tuning-pipeline>
- Zenodo: DOI will be added after the first archived release.

Add the version DOI after publishing the archived release on Zenodo, and update
the same value in `CITATION.cff`.

Suggested manuscript text:

> **Code Availability.** Our source code is available on GitHub and Zenodo as
> follows: [GitHub URL] and [Zenodo DOI].

## Citation

GitHub reads [`CITATION.cff`](CITATION.cff) and exposes the repository citation
through its “Cite this repository” menu. The software DOI will be added after
the versioned Zenodo archive is published.

## License

The source code is released under the MIT License. Dataset and base-model
licenses remain applicable separately.
