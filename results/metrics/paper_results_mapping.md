# Paper results mapping

| Paper item | Artifact or generation source |
|---|---|
| Tables 3–5, main results | `results/predictions/<dataset>/*.csv` → `scripts/evaluate.py`; NEU Dual + Pipeline uses r=16/64 |
| Table 6 / Figure 4, ablation | Base, Masking, and Full prediction/metric files for the three reported ranks |
| Table 7, minority recall | Classification reports generated from the same ablation predictions |
| Table 8 / Figure 5, Joint ranks | `configs/rank_sweeps/joint_rank_sweep.yaml` and audited rank metrics |
| Table 9, Dual ranks | `configs/rank_sweeps/dual_rank_sweep_neu.yaml` and audited rank metrics |
| Table 10, efficiency | `performance.json` from `scripts/predict.py` on UIT-VSFC/A100 40 GB |
| Figure 6, confusion matrices | NEU-ESC r=32 predictions → `scripts/generate_figures.py` |
| Figure 7, error analysis | NEU-ESC r=32 predictions → `scripts/generate_figures.py` |

Files containing published numerical values should be added only after verifying
them against the final manuscript tables.
