# Wav2Pwn Dataset Diversity Evaluation

This standalone project runs the main Wav2Pwn evaluation independently on
`LibriSpeech`, `CommonVoice_wav`, and `VoxPopuli`. It is intentionally
separated from the earlier surrogate-model pipeline so each dataset can produce
its own adversarial examples, probing structures, transfer matrices, and
family-level clustering analysis without mixing intermediate artifacts.

## Research Goal

For each dataset independently, the pipeline:

1. generates adversarial examples for each source SSL-ASR model,
2. probes model relationships on clean audio,
3. measures cross-model transfer attack behavior,
4. builds transfer and probing similarity matrices,
5. computes family-level structure metrics for Table III.

The independent-per-dataset design matters because reviewer concerns about
dataset diversity are not answered by pooling all audio together. Instead, we
show whether transfer structures and family-level probing relationships remain
stable when the speech source changes.

## Directory Layout

```text
wav2pwn_dataset_diversity_eval/
├── README.md
├── requirements.txt
├── configs/
│   ├── base.yaml
│   └── models.yaml
├── run_dataset_experiment.py
├── build_table3_summary.py
└── src/wav2pwn_diversity/
    ├── __init__.py
    ├── attacks.py
    ├── config.py
    ├── data.py
    ├── families.py
    ├── logging_utils.py
    ├── metrics.py
    ├── models.py
    ├── probing.py
    ├── transfer.py
    └── visualize.py
```

## Supported Model Families

The default model registry covers approximately nine SSL-ASR systems:

- Wav2Vec2:
  - `facebook/wav2vec2-base-100h`
  - `facebook/wav2vec2-large-960h`
  - `facebook/wav2vec2-large-robust-ft-libri-960h`
- HuBERT:
  - `facebook/hubert-large-ls960-ft`
  - `facebook/hubert-xlarge-ls960-ft`
- WavLM:
  - `microsoft/wavlm-base-plus`
  - `microsoft/wavlm-large`
- Conformer/Rope:
  - `facebook/wav2vec2-conformer-rope-large-100h-ft`
- VoxPopuli fine-tuned:
  - `facebook/wav2vec2-base-10k-voxpopuli-ft-en`

## Dataset-Independent Outputs

Each dataset produces its own outputs under `results/<dataset_slug>/`:

- `transfer_matrix_<dataset>.csv`
- `transfer_tasr_matrix_<dataset>.csv`
- `transfer_uasr_matrix_<dataset>.csv`
- `probing_matrix_<dataset>.csv`
- `transfer_pairs_<dataset>.csv`
- `probing_pairs_<dataset>.csv`
- `family_metrics_<dataset>.json`
- heatmaps and clustering figures

The global Table III summary is exported as:

- `results/table3_summary.csv`
- `results/table3_summary.json`

## Usage

Activate your environment and run one dataset at a time.

```bash
conda activate /path/to/your/env
cd wav2pwn_dataset_diversity_eval
python run_dataset_experiment.py --dataset librispeech --config configs/base.yaml
python run_dataset_experiment.py --dataset commonvoice --config configs/base.yaml
python run_dataset_experiment.py --dataset voxpopuli --config configs/base.yaml
python build_table3_summary.py --config configs/base.yaml
```

## Notes

- The default sample budget is `2000` per dataset and is configured in
  `configs/base.yaml`.
- This pipeline assumes each dataset is processed independently. It does not
  combine audio across datasets.
- Transfer structures are important because they reveal which model families
  share similar adversarial vulnerabilities.
- Probing is used to capture family-level relationships before attack time. If
  these relationships remain stable across datasets, that is evidence of robust
  structure rather than dataset-specific noise.
