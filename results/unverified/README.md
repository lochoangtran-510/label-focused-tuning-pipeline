# Unverified imported artifacts

This directory is intentionally excluded from the public Git repository except
for this notice. It contains prediction and performance files that arrived
inside the uploaded checkpoint folders. They are kept locally for audit but
must not be used as paper evidence until their provenance and numbers have been
checked against the final manuscript.

In particular, the imported NEU-ESC files are named as a Dual Adapter `r=16/16`
run but arrived inside the released `r=16/64` Sentiment adapter directory. They
were separated rather than incorrectly relabeled as `r=16/64`.

Three additional supplied metric reports are quarantined:

- `neu_esc/dual_pipeline_r64_r16/incomplete_report.txt` is missing its fallback
  summary and the beginning of the Sentiment section.
- `uit_vsfc/dual_base_r8_r16/report_mismatch.txt` reports Topic Accuracy/mF1
  89.64/81.72 rather than the final paper values 89.70/81.76.
- `uit_vsfc/dual_pipeline_r8_r16/duplicate_of_dual_base.txt` is byte-identical
  to that mismatching Dual Base report, so it cannot substantiate Dual +
  Pipeline.
