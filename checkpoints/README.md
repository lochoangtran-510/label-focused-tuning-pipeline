# LoRA checkpoints

The Vistral-7B base model is not redistributed. This directory is reserved for
the six principal LoRA releases used in the main comparison:

```text
checkpoints/
├── neu_esc/{joint_pipeline_r32,dual_pipeline}/
├── uit_vsfc/{joint_pipeline_r8,dual_pipeline}/
└── victsd/{joint_pipeline_r8,dual_pipeline}/
```

Joint directories contain one PEFT adapter. Dual directories contain independent
`sentiment/` and `topic/` adapters. The trained files are not currently present
in this workspace and must not be represented by fabricated placeholders. Add
the verified adapters to the Zenodo release (or link them from this directory)
after reproducing and auditing each run.
