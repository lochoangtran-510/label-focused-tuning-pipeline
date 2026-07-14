# Zenodo release checklist

This checklist defines the contents of the reproducibility release. It should
be reviewed before publishing the Zenodo draft because a published version
cannot be silently replaced.

## Release contents

Upload two versioned archives:

1. `label-focused-tuning-source-v1.0.0.zip`
   - public source code, configs, tests, documentation, and aggregate metrics;
   - excludes the Git history, local environments, caches, raw/processed data,
     exploratory notebooks, manuscript PDF, checkpoints, predictions, and
     generated figures.
2. `label-focused-tuning-adapters-v1.0.0.zip`
   - the canonical `checkpoints/` directory;
   - includes 37 verified PEFT adapters and `MANIFEST.sha256`;
   - excludes raw source ZIP files, row-level predictions, and figures.

The Vistral-7B base model and the three datasets are not redistributed.

## Known checkpoint gaps

Two task-specific adapters were not retained and are not included:

- NEU-ESC Dual Pipeline Topic `r=16`;
- ViCTSD Dual Base Topic `r=32`.

Source archives bearing those names were found to contain a different LoRA
rank, so they are deliberately excluded. The training configs needed to
regenerate the missing adapters remain public.

## Integrity and privacy checks

- Verify all released weights from inside the extracted checkpoint tree:

  ```bash
  sha256sum -c MANIFEST.sha256
  ```

- Confirm that neither archive contains `.env`, credentials, Hugging Face
  tokens, private Google Drive paths, exploratory notebooks, the manuscript
  PDF, row-level predictions, or generated figures.
- Keep Hugging Face credentials in Colab Secrets or environment variables; do
  not place them in notebooks or command history included in the release.

## Suggested Zenodo metadata

- **Title:** Label-Focused Tuning Pipeline
- **Resource type:** Software
- **Version:** 1.0.0
- **Reserved version DOI:** `10.5281/zenodo.21348518`
- **Creators:** Hoang Xuan Loc Tran; Thanh Hung Bui
- **Licenses:** MIT for the source-code archive; Academic Free License 3.0
  (`AFL-3.0`) for the adapter archive, following the license declared by the
  Vistral-7B-Chat base model
- **Related identifier:**
  `https://github.com/lochoangtran-510/label-focused-tuning-pipeline`
- **Description:** Source code, reproducible configurations, aggregate metrics,
  and audited PEFT/LoRA adapters for the associated article.

Save the upload as a draft first. After checking the file list, metadata,
creator order, license, reserved DOI, and known-gap statement, publish it. The
reserved version DOI above has already been added to `README.md` and
`CITATION.cff`; it becomes registered only when the record is published.
