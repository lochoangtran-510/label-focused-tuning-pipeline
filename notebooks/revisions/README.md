# Corrected standalone Colab notebooks

These notebooks publish the corrected working implementations with execution
outputs cleared. They are opt-in revision runs, not regenerated paper results.
Open a notebook in Colab, select a GPU runtime, and edit its configuration cell
before running top to bottom. Model training has not been rerun as part of this
repository update.

## Training entry points

| Dataset | Objective | Alpha scope | Notebook |
|---|---|---|---|
| UIT-VSFC | CE + completion masking + alpha | Terminal only | [terminal](uit_vsfc/uit_vsfc_mask_alpha_ce_terminal_only_r8_len1024.ipynb) |
| UIT-VSFC | CE + completion masking + alpha | All label tokens | [all tokens](uit_vsfc/uit_vsfc_mask_alpha_ce_all_label_tokens_r8_len1024.ipynb) |
| NEU-ESC | CE + completion masking + alpha | Terminal only | [terminal](neu_esc/neu_mask_alpha_ce_terminal_only_r8_len2024.ipynb) |
| NEU-ESC | CE + completion masking + alpha | All label tokens | [all tokens](neu_esc/neu_mask_alpha_ce_all_label_tokens_r8_len2024.ipynb) |
| ViCTSD | CE + completion masking + alpha | Terminal only | [terminal](victsd/victsd_mask_alpha_ce_terminal_only_r8_len1024.ipynb) |
| ViCTSD | CE + completion masking + alpha | All label tokens | [all tokens](victsd/victsd_mask_alpha_ce_all_label_tokens_r8_len1024.ipynb) |
| ViCTSD | Base CE (prompt + response) | Disabled | [base CE](victsd/victsd_base_ce_r8_len1024.ipynb) |

All seven use LoRA rank 8, LoRA alpha 16, dropout 0.05, learning rate 1e-4,
batch size 6, accumulation 4, and seed 42. UIT-VSFC and ViCTSD use maximum length
1024 and a 3-epoch schedule. NEU-ESC uses maximum length 2024 and a 2800-step
schedule. Early stopping and best-checkpoint selection use validation loss.
The new NEU schedule is distinct from the historical paper's r=32 ablation.

The terminal/all notebooks for each dataset share the same training
implementation; `ALPHA_SCOPE` selects the weighting rule. Other completion
tokens retain weight 1. Every notebook audits the mask and alpha before training.
Base CE applies no completion mask, alpha, or focal modulation; padding is
ignored through the attention mask, so genuine EOS tokens remain supervised.

The masked notebooks retain the reference TRL completion collator. When
`pad_token_id == eos_token_id`, that collator also ignores genuine EOS tokens.
Both alpha scopes share this policy, so their comparison changes only alpha
placement. Comparing the published base CE notebook against a masked notebook
also changes EOS supervision; it is not a pure prompt-mask ablation.

## Data paths and outputs

The example Drive root is `/content/drive/MyDrive/label-focused-tuning/`, with
dataset folders `uit_vsfc`, `neu_esc`, and `victsd`. Edit the CSV paths in the
configuration cell to match your own copies of the official splits. The
notebooks do not redistribute data or contain access tokens. For gated model or
dataset access, use the interactive Hugging Face login or a private Colab secret.

ViCTSD accepts the raw Title/Comment columns and combines them using the source
notebook's `[SEP]` convention. Preserve the configured column names, label maps,
prompt, and split membership when comparing runs. Training length filtering is
performed before class alpha is computed; validation truncation is rejected.

Run names include the objective, alpha scope, and seed. Choose a fresh run
directory when repeating an experiment. Keep each notebook's training manifest
with its adapter and results. Do not use the test split to select alpha scope or
training checkpoints.

## Evaluation helpers

- [ViCTSD full-test notebook](victsd/victsd_test_ablation_fixed.ipynb) and
  [script](../../scripts/revisions/victsd_test_ablation_fixed.py): evaluate one
  selected adapter on the complete test split with a strict JSON parser. Invalid
  task predictions remain sentinel errors in the metrics; valid fields are
  preserved when another field is invalid. Edit the adapter and CSV paths. This
  revision evaluator differs from the shared CLI's train-majority fallback
  policy; label its results separately.
- [NEU three-checkpoint validation helper](../../scripts/revisions/neu_val_mf1_3_checkpoints.py):
  run inside the NEU training notebook session to compare three adapters on
  validation macro-F1, using train-majority fallback. It depends on that
  session's variables and is not a standalone test-set evaluator.

Standalone notebooks preserve their dataset-specific prompts and training
protocols. Shared CLI presets use the repository's shared prompt builders. Treat
these as separate entry points; identical method names alone do not establish
identical prompts or byte-identical training inputs.

## Provenance

[`PROVENANCE.json`](PROVENANCE.json) records the original corrected working
notebook filenames and SHA-256 hashes, plus the hashes of the published copies.
Publication changes clear outputs and execution metadata, replace personal
example Drive paths, and derive terminal-only counterparts by changing
`ALPHA_SCOPE`. Some in-notebook provenance constants intentionally retain the
original working-source hashes. Use the published hashes in this JSON to verify
the files in this repository.

No historical output metrics, private exploratory notebooks, credentials, raw
datasets, or checkpoint files were imported with these notebooks. Zenodo v1.0.0
remains the earlier archived snapshot.
