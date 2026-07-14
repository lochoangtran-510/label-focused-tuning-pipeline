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

The uploaded reports were checked against the headline values in the revised
manuscript. All 36 aggregate reports are now present. Values are transcribed
from supplied full-test outputs; missing results must never be reconstructed by
editing metrics.

The NEU-ESC Dual + Pipeline r=64/16 report was supplied in two parts: a raw
valid-only runtime report covering 6,610 rows and a truncated full-test
reevaluation covering all 6,613 rows after fallback. Its canonical report
combines the runtime fields from the former with the corrected classification
metrics from the latter and clearly records this reconciliation.

The corrected UIT-VSFC Dual + Pipeline r=8/16 report has Topic Accuracy 90.27%
(not 92.70%), Topic Macro F1 81.26%, a runtime of 116.3 seconds, and zero parse
errors. These are the values from the revised manuscript confirmed by the
author.

The corrected UIT-VSFC Dual Base r=8/16 report has Topic Accuracy 89.64%,
Topic Macro F1 81.72%, a runtime of 117.1 seconds, and zero parse errors. These
are also author-confirmed values from the revised manuscript.

`uit_vsfc/table10_efficiency.json` and `.csv` consolidate the five UIT-VSFC
efficiency rows from their retained raw reports. Table 10 uses Joint r=16 and
Dual Sentiment/Topic r=8/16; throughput denotes generated output tokens per
second.

Row-level Joint + Pipeline r=8 predictions were also supplied for UIT-VSFC and
ViCTSD. Their row counts, label sets, Accuracy, Macro F1, and Weighted F1 were
recomputed and matched the corresponding reports, so they are retained under
`results/predictions/<dataset>/joint_pipeline_r8/`. Other aggregate reports do
not currently have verified row-level prediction files.
