# LoRA checkpoints

The Vistral-7B base model is not redistributed. The current Zenodo checkpoint
tree contains the principal configurations for all datasets, the complete
UIT-VSFC ablation and Joint rank-sensitivity adapters, and all verified NEU-ESC
ablation/rank adapters received so far:

```text
checkpoints/
├── neu_esc/
│   ├── joint_base_{r8,r16,r32,r64}/
│   ├── joint_masking_r32/
│   ├── joint_pipeline_{r8,r16,r32,r64}/
│   ├── dual_base_r16_r64/{sentiment_r16,topic_r64}/
│   └── dual_pipeline/{sentiment_r*,topic_r*}/
├── uit_vsfc/
│   ├── joint_base_{r8,r16,r32,r64}/
│   ├── joint_masking_r8/
│   ├── joint_pipeline_{r8,r16,r32,r64}/
│   ├── dual_base_r8_r16/{sentiment_r8,topic_r16}/
│   └── dual_pipeline_r8_r16/{sentiment_r8,topic_r16}/
└── victsd/
    ├── joint_base_r8/
    ├── joint_masking_r8/
    ├── joint_pipeline_r8/
    ├── dual_base_r8_r32/sentiment_r8/
    └── dual_pipeline_r8_r32/{sentiment_r8,topic_r32}/
```

Joint directories contain one PEFT adapter. Dual directories contain independent
task adapters and encode both ranks in the parent directory name. Each adapter
contains its configuration, SafeTensors weights, and tokenizer files.
Dataset-level README files map the adapters to the paper tables, and
`MANIFEST.sha256` records every verified weight checksum. The NEU-ESC
Topic Pipeline r=16 adapter is intentionally absent until the mislabeled source
archive is replaced with a verified rank-16 checkpoint.

The ViCTSD Dual Base Topic r=32 adapter is likewise absent because the source
archive carrying that name was verified to contain a rank-64 adapter.

Checkpoint binaries are excluded from Git because several files exceed GitHub's
normal file-size limit. Publish the complete `checkpoints/` tree in the versioned
Zenodo archive and keep this layout unchanged so the README inference commands
remain unambiguous.

The adapters require the separately downloaded
`Viet-Mistral/Vistral-7B-Chat` base model. The Zenodo record should identify
the adapter archive under the Academic Free License 3.0 (`AFL-3.0`) declared by
that base model; the repository source code remains under the MIT license.
