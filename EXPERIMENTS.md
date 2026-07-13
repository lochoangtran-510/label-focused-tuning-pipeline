# Experiments reported in the paper

This file distinguishes reusable implementation capabilities from experiments
that were actually reported. Do not interpret support for a configuration as a
claim that it was evaluated in the article.

All public presets use `epochs: 3` and `max_steps: -1`.

## Main comparison

Tables 3–5 report Vistral-7B Zero-shot, Joint LoRA, Joint + Pipeline, Dual
Adapter, and Dual + Pipeline results on NEU-ESC, ViCTSD, and UIT-VSFC alongside
external baselines. Results marked with a dagger in the paper came from prior
work and were not re-evaluated in this repository.

In the NEU-ESC main comparison, Dual + Pipeline uses the asymmetric
Sentiment/Topic rank configuration `r=16/64`. The complete Dual rank analysis
remains listed separately in Table 9.

The zero-shot Vistral baseline is reproduced through `scripts/predict.py` with
`--architecture zero_shot`; it uses the dataset-specific two-task instruction
without a one-shot exemplar and does not load a LoRA adapter.
Task-specific majority fallback retains the complete official test split; the
paper reports fallback for 15/6,613 NEU-ESC, 15/1,000 ViCTSD, and 18/3,166
UIT-VSFC zero-shot samples.

## Joint ablation (Table 6)

Only three configurations were reported:

| Dataset | Rank | Base | + Masking | Full Pipeline |
|---|---:|---:|---:|---:|
| NEU-ESC | 32 | Yes | Yes | Yes |
| UIT-VSFC | 8 | Yes | Yes | Yes |
| ViCTSD | 8 | Yes | Yes | Yes |

There is no standalone Focal-only row in Table 6.

## Rank sensitivity

- Joint architecture (Table 8): ranks 8, 16, 32, and 64 on NEU-ESC and
  UIT-VSFC only, comparing Base with Full Pipeline.
- Dual Adapter + Pipeline (Table 9): NEU-ESC only, with Sentiment/Topic rank
  pairs `8/8`, `16/16`, `32/32`, `64/64`, `16/64`, `64/16`, and `32/8`.
- No rank-sensitivity table was reported for ViCTSD.

## Additional analysis

- Inference efficiency (Table 10): UIT-VSFC test set only, on an A100 40 GB.
- Minority-class recall analysis: all three datasets using their ablation
  configurations.
- Detailed error analysis and confusion matrices: NEU-ESC at r=32.

For Table 9, every Dual configuration is evaluated on all 6,613 NEU-ESC test
samples. Three samples per configuration require at least one task-specific
fallback; no row is removed from metric computation.
