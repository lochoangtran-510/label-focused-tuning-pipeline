# Paper results mapping

| Paper item | Artifact or generation source |
|---|---|
| Table 3, NEU-ESC | `neu_esc/{zero_shot,joint_base/r32,joint_pipeline/r32,dual_base/r16_r64,dual_pipeline/r16_r64}` |
| Table 4, ViCTSD | `victsd/{zero_shot,joint_base/r8,joint_pipeline/r8,dual_base/r8_r32,dual_pipeline/r8_r32}` |
| Table 5, UIT-VSFC | `uit_vsfc/{zero_shot,joint_base/r16,joint_pipeline/r16,dual_base/r8_r16,dual_pipeline/r8_r16}` |
| Table 6 / Figure 4, ablation | `joint_base`, `joint_masking`, and `joint_pipeline` at NEU r=32, UIT r=8, and ViCTSD r=8 |
| Table 7, minority recall | Classification reports in the same ablation files |
| Table 8 / Figure 5, Joint ranks | NEU-ESC and UIT-VSFC `joint_base` and `joint_pipeline`, r=8/16/32/64 |
| Table 9, Dual ranks | NEU-ESC `dual_pipeline`, including the reconciled full-test r=64/16 report |
| Table 10, efficiency | `uit_vsfc/table10_efficiency.{json,csv}` consolidates Zero-shot, Joint r=16, and Dual r=8/16 timing fields; Dual Base records 117.1 s/588 tok/s/27.0 samples/s and Dual + Pipeline records 116.3 s/593 tok/s/27.2 samples/s |
| Figure 6, confusion matrices | Requires row-level NEU-ESC r=32 predictions and `scripts/generate_figures.py` |
| Figure 7, error analysis | Requires row-level NEU-ESC r=32 predictions and `scripts/generate_figures.py` |

Paths above are relative to `results/metrics/` and omit the `.txt` suffix. Text
reports support aggregate tables, but do not replace `predictions.csv` for
regenerating confusion matrices or sample-level error analysis.

Verified row-level predictions currently exist only for UIT-VSFC and ViCTSD
Joint + Pipeline r=8 under `results/predictions/`.
