# Reproduction artifacts

Prediction CSVs and metrics are kept separate from model inference so tables and
figures can be regenerated without loading Vistral. No numerical result is
fabricated in this repository: populate these directories only with audited
outputs from the released checkpoints.

- `predictions/`: full official test splits, including raw output and fallback flags.
- `metrics/`: table-level CSV/JSON artifacts.
- `figures/`: generated plots.
- `logs/`: run logs and environment metadata.
