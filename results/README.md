# Reproduction artifacts

Prediction CSVs and metrics are kept separate from model inference so tables and
figures can be regenerated without loading Vistral. No numerical result is
fabricated in this repository: populate these directories only with audited
outputs from the released checkpoints.

- `predictions/`: verified row-level outputs on complete official test splits.
- `metrics/`: aggregate full-test TXT/JSON reports organized by dataset,
  architecture, and LoRA rank.
- `figures/`: generated plots.
- `logs/`: run logs and environment metadata.

`unverified/` is a quarantine area for imported legacy artifacts whose filename
or provenance does not match a released checkpoint. Its contents are excluded
from Git and must not be cited until audited.

See `metrics/README.md` for the artifact layout, manuscript mapping, and the
specific reports that still need to be recovered or rerun.
