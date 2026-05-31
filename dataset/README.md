# Dataset Folder

This folder is kept as a placeholder for datasets referenced by the paper.

No raw datasets are distributed in this GitHub repository. Reviewers and users
should place their local copies here before running the experiments.

Suggested layout:

```text
dataset/
├── LibriSpeech_wav/
├── CommonVoice_wav/
└── voxpopuli/
```

Notes:

- `LibriSpeech_wav/` is used by the dataset-diversity evaluation pipeline.
- `CommonVoice_wav/` is used by the dataset-diversity evaluation pipeline.
- `voxpopuli/` is used by the dataset-diversity evaluation pipeline.
- Some legacy scripts may also expect a separate `LibriSpeech/` path outside
  this folder, as documented in their script-level usage.
