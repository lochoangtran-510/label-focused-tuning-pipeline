# Changelog

## 2026-10-16 — corrected training source and revision notebooks

- Assign class alpha by JSON field, label, and token span instead of a global
  terminal-token lookup. This supports shared token IDs across labels and
  tokenization that depends on the surrounding JSON.
- Separate completion masking, class alpha, and focal loss. Keep terminal-only
  alpha as the default; add all-label-token alpha as an explicit experiment.
- Compute per-task balanced alpha from exactly the length-filtered training rows.
  Keep loss normalization as the mean over active causal target tokens.
- Preserve true EOS labels when padding uses the same token ID as EOS. Base CE
  includes prompt and response loss; completion masking excludes prompt loss.
- Record data, tokenizer, seed, class counts, alpha scope, and objective semantics
  in training manifests. Seed LoRA initialization and data ordering, reject
  truncated validation examples, and prevent accidental run-directory reuse.
- Add opt-in revision configs and seven clean standalone training notebooks for
  UIT-VSFC, NEU-ESC, and ViCTSD, including ViCTSD base CE. Include ViCTSD test
  evaluation and the NEU validation-only checkpoint comparison helper.
- Update training dependency pins and add CPU regression tests.

This update does not modify archived result files, checkpoint checksums, the
paper's historical experiment matrix, citation metadata, or the published Zenodo
v1.0.0 record. The corrected source requires new training runs before any new
reproduction or performance claim can be made.
