# Experiments reported in the paper

This file distinguishes reusable implementation capabilities from experiments
that were actually reported. Do not interpret support for a configuration as a
claim that it was evaluated in the article.

Historical paper presets use `epochs: 3` and `max_steps: -1`. Current corrected
source is not a claim that the archived paper results have been rerun; the
historical metrics and released adapters remain unchanged.

## Main comparison

Tables 3–5 report Vistral-7B Zero-shot, Joint LoRA, Joint + Pipeline, Dual
Adapter, and Dual + Pipeline results on NEU-ESC, ViCTSD, and UIT-VSFC alongside
external baselines. Results marked with a dagger in the paper came from prior
work and were not re-evaluated in this repository.

In the NEU-ESC main comparison, Dual + Pipeline uses the asymmetric
Sentiment/Topic rank configuration `r=16/64`. The complete Dual rank analysis
remains listed separately in Table 9.

The zero-shot Vistral baseline can be run through `scripts/predict.py` with
`--architecture zero_shot`; it uses the dataset-specific two-task instruction
without a one-shot exemplar and does not load a LoRA adapter.
The archived paper reports used task-specific majority fallback while retaining
the complete official test split: 15/6,613 NEU-ESC, 15/1,000 ViCTSD, and
18/3,166 UIT-VSFC zero-shot samples. These are historical archive counts, not
measurements from the current no-fallback code. The retained reports and scores
have not been changed or recomputed.

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

For the archived Table 9 reports, every Dual configuration was evaluated on all
6,613 NEU-ESC test samples. Those reports record at least one task-specific
fallback on three samples per configuration; no row was removed from metric
computation. This describes the historical evaluation, not the current policy.

## Current prediction and evaluation policy

For all three datasets, current code keeps every evaluation row and uses no
majority-label fallback. Invalid task predictions are recorded as `PARSE_ERROR`
and counted as wrong; a valid prediction for the other task remains valid.
Macro-F1 and weighted F1 use each task's fixed dataset label set, with no extra
target class for the sentinel. Compatibility names ending in `*_corrected`
refer to normalized scoring values, without class imputation.

The NEU validation helper follows this no-fallback scoring policy while keeping
its existing lenient JSON extraction. The ViCTSD revision evaluator keeps its
existing strict JSON parser. Compare runs using the same parser and scoring
policy. This update does not infer new invalid-output counts or regenerate any
archived paper metrics.

## New revision experiments (2026-10-16)

These opt-in configurations are separate from the reported tables above. No new
scores are included in this source update.

| Dataset | Rank | Maximum length | Schedule | Added controls |
|---|---:|---:|---|---|
| UIT-VSFC | 8 | 1024 | 3 epochs | Masked CE + alpha (terminal/all); masked focal without alpha |
| NEU-ESC | 8 | 2024 | 2800 steps | Masked CE + alpha (terminal/all); masked focal without alpha |
| ViCTSD | 8 | 1024 | 3 epochs | Base CE; masked CE + alpha (terminal/all); masked focal without alpha |

Run presets from `configs/additional/`. Standalone Colab notebooks and their
data-path instructions are in [`notebooks/revisions`](notebooks/revisions/README.md).
The NEU-ESC revision uses `protocol: revision`; it must not be mixed with the
historical r=32, length-1024 paper ablation.

For the terminal-versus-all comparison, hold the seed, model, train/validation
splits, length filter, prompt, optimizer, and checkpoint-selection rule constant.
The reference seed is 42. Alpha comes only from the retained training rows,
separately per task. Validation selects checkpoints within a run; test results
must not be used to choose the alpha scope. Report single-seed findings as such.

Changing alpha scope changes how many label subtokens receive the class weight;
it does not change the completion mask or the loss denominator. In both scopes,
unweighted completion tokens (including JSON syntax) still contribute CE. Base
CE also includes the prompt, and ignores only padding. Current source rejects
truncated validation responses, ambiguous label spans, and existing output runs
rather than silently changing the experiment.
