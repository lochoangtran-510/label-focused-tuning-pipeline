# Unverified imported artifacts

This directory is intentionally excluded from the public Git repository except
for this notice. It contains prediction and performance files that arrived
inside the uploaded checkpoint folders. They are kept locally for audit but
must not be used as paper evidence until their provenance and numbers have been
checked against the final manuscript.

In particular, the imported NEU-ESC files are named as a Dual Adapter `r=16/16`
run but arrived inside the released `r=16/64` Sentiment adapter directory. They
were separated rather than incorrectly relabeled as `r=16/64`.

The NEU-ESC r=64/16 source artifacts are retained here for provenance:

- `neu_esc/dual_pipeline_r64_r16/incomplete_report.txt` is missing its fallback
  summary and the beginning of the Sentiment section.
- `neu_esc/dual_pipeline_r64_r16/raw_pre_fallback_report.txt` contains runtime
  metrics but calculates task metrics on only 6,610 valid rows. The canonical
  report under `results/metrics/` reconciles these sources and evaluates all
  6,613 rows after majority fallback.

The original `uit_vsfc/dual_base_r8_r16/report_mismatch.txt` was quarantined
because it differed from an older PDF. The author subsequently confirmed that
89.64% Topic Accuracy and 81.72% Topic Macro F1 are the corrected manuscript
values, so a canonical report is now stored under `results/metrics/`.

The earlier
`uit_vsfc/dual_pipeline_r8_r16/duplicate_of_dual_base.txt` remains here only as
an audit trail: it was an accidental duplicate. The corrected Dual + Pipeline
report is now stored under `results/metrics/`.
