# Metric reports

These files are full-test evaluation reports supplied for the final manuscript.
They contain aggregate metrics and classification reports, not row-level model
predictions. Dataset and method names use the same identifiers as the public
configs. For Dual Adapter reports, ranks are ordered as Sentiment/Topic.

```text
metrics/<dataset>/
├── zero_shot/report.txt
├── joint_base/r<rank>.txt
├── joint_masking/r<rank>.txt
├── joint_pipeline/r<rank>.txt
├── dual_base/r<sentiment>_r<topic>.txt
└── dual_pipeline/r<sentiment>_r<topic>.txt
```

The uploaded reports were checked against the headline values in Tables 3–9 of
the final manuscript. Three source files were quarantined under
`results/unverified/` and are intentionally absent here:

- NEU-ESC Dual + Pipeline `r=64/16`: the report is truncated before the
  Sentiment summary and contains no fallback summary.
- UIT-VSFC Dual Base `r=8/16`: the uploaded report gives Topic Accuracy/mF1 of
  89.64/81.72, whereas Table 5 reports 89.70/81.76.
- UIT-VSFC Dual + Pipeline `r=8/16`: the uploaded file is byte-for-byte
  identical to that mismatching Dual Base report and does not match the Dual +
  Pipeline row in Table 5 or Table 10.

Do not reconstruct missing values by editing these reports. Recover the original
outputs or rerun inference from the corresponding adapters.

Row-level Joint + Pipeline r=8 predictions were also supplied for UIT-VSFC and
ViCTSD. Their row counts, label sets, Accuracy, Macro F1, and Weighted F1 were
recomputed and matched the corresponding reports, so they are retained under
`results/predictions/<dataset>/joint_pipeline_r8/`. Other aggregate reports do
not currently have verified row-level prediction files.
