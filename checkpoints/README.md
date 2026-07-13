# LoRA checkpoints

The Vistral-7B base model is not redistributed. The Zenodo checkpoint package
uses the following layout for the six principal configurations (nine PEFT
adapters):

```text
checkpoints/
├── neu_esc/
│   ├── joint_pipeline_r32/
│   └── dual_pipeline_r16_r64/{sentiment_r16,topic_r64}/
├── uit_vsfc/
│   ├── joint_pipeline_r8/
│   └── dual_pipeline_r8_r16/{sentiment_r8,topic_r16}/
└── victsd/
    ├── joint_pipeline_r8/
    └── dual_pipeline_r8_r32/{sentiment_r8,topic_r32}/
```

Joint directories contain one PEFT adapter. Dual directories contain independent
task adapters and encode both ranks in the parent directory name. Each adapter
contains its configuration, SafeTensors weights, tokenizer files, and a concise
model card. `MANIFEST.sha256` records the weight checksums.

Checkpoint binaries are excluded from Git because several files exceed GitHub's
normal file-size limit. Publish the complete `checkpoints/` tree in the versioned
Zenodo archive and keep this layout unchanged so the README inference commands
remain unambiguous.
