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
├── notebooks/label_focused_pipeline.ipynb  # recommended entry point
├── scripts/predict.py                      # Joint/Dual vLLM inference
├── scripts/evaluate.py                     # full-test majority fallback metrics
├── scripts/train.py                        # YAML-driven training entry point
├── configs/                                # reported reproduction presets
├── src/label_focused/
│   ├── datasets.py                         # schemas, labels, prompts, loaders
│   ├── prompting.py                        # unified chat/JSON prompt engine
│   ├── training.py                         # LoRA, masking, focal loss, trainer
│   └── evaluation.py                       # classification metrics
├── data/README.md                          # expected local data layout
├── EXPERIMENTS.md                          # exact scope of reported experiments
├── requirements.txt
└── CITATION.cff
```

The original exploratory notebooks are retained temporarily for result auditing;
the clean notebook above is the public and reproducible entry point.
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

Prepare the datasets as described in [`data/README.md`](data/README.md), then
open `notebooks/label_focused_pipeline.ipynb`. Select a dataset in the
configuration cell:

```python
DATASET_NAME = "neu_esc"  # "neu_esc", "uit_vsfc", or "victsd"
```

No Hugging Face token is stored in the source. Authenticate outside the
notebook only when access to a selected model requires it.

## YAML-driven training

All public reproduction presets train for **3 complete epochs**. They set
`max_steps: -1`, so `TrainingArguments` does not replace the epoch schedule with
a fixed step limit.

Run a reported configuration with:

```bash
python scripts/train.py \
  --config configs/ablation/neu_esc_full_r32.yaml
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
  --joint-adapter outputs/neu_esc/joint/pipeline_r32/best_model \
  --output-dir outputs/neu_esc/joint/pipeline_r32/inference
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
The ablation reported only Base, Masking, and Full Pipeline; a standalone
Focal-only configuration was not reported as a separate row. Outputs are written
under `outputs/<dataset>/<architecture>/<configuration>/` and excluded from Git.

## Data availability

The datasets are not redistributed here. Please obtain and cite them from their
original publications:

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

## License

The source code is released under the MIT License. Dataset and base-model
licenses remain applicable separately.
