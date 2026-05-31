# Black-Box Surrogate ASR Attack Pipeline

This folder is a standalone research pipeline for surrogate-based black-box adversarial attacks against ASR systems. It is intentionally separated from the older repository contents so the extraction, surrogate training, white-box optimization, and black-box transfer experiments can be reproduced without entangling legacy scripts.

## Threat model

The target ASR system is `facebook/hubert-large-ls960-ft`, treated as a black box. We assume the attacker can submit audio queries and read back transcriptions, but cannot access gradients, logits, hidden states, or parameters. The workflow is:

1. Query the black-box HuBERT target on a configurable subset of LibriSpeech `train-clean-100`.
2. Save pseudo-labels `(audio_path, hubert_transcription, duration)`.
3. Train a surrogate `facebook/wav2vec2-base-100h` CTC model on those pseudo-labels.
4. Run white-box PGD on the surrogate only.
5. Transfer the adversarial waveform back to the black-box HuBERT target.
6. Report clean CER, targeted attack success rate, untargeted attack success rate, and attack cost.

## Project structure

```text
blackbox_surrogate_asr/
├── configs/base.yaml
├── src/asr_blackbox/
├── query_target.py
├── train_surrogate.py
├── evaluate_surrogate.py
├── pgd_attack.py
├── transfer_eval.py
├── scripts/run_query_efficiency.py
├── scripts/export_paper_tables.py
├── requirements.txt
└── README.md
```

## Environment

```bash
cd blackbox_surrogate_asr
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The intended query dataset is LibriSpeech `train-clean-100`. The current repository already contains `../LibriSpeech/test-clean`, but not necessarily `train-clean-100`, so pass the path explicitly when you have the full training split.

Pseudo-label JSON format:

```json
[
  {
    "sample_id": "query_000000",
    "audio_path": "/abs/path/to/audio.flac",
    "transcription": "THE WEATHER IS GOOD TODAY",
    "duration": 3.12,
    "ground_truth": "THE WEATHER IS GOOD TODAY"
  }
]
```

## Step 1: Query the black-box target

```bash
python query_target.py \
  --dataset-root ../LibriSpeech/train-clean-100 \
  --num-samples 2000 \
  --output-json artifacts/pseudo_labels.json
```

Supported query-budget experiments are configured for `100`, `500`, `1000`, `2000`, and `5000`.

## Step 2: Train the surrogate

The surrogate training script uses Hugging Face `Trainer`, CTC training, mixed precision, gradient accumulation, checkpointing, and TensorBoard logging.

```bash
python train_surrogate.py \
  --config configs/base.yaml \
  --pseudo-labels artifacts/pseudo_labels.json
```

TensorBoard logs are written under `logs/tensorboard`.

## Step 3: Evaluate imitation quality

This compares surrogate outputs against HuBERT pseudo-label outputs and reports WER/CER.

```bash
python evaluate_surrogate.py \
  --pseudo-labels artifacts/pseudo_labels.json \
  --surrogate-model checkpoints/best_model
```

## Step 4: White-box PGD on the surrogate

The PGD attack is performed only on the surrogate model. This is the white-box optimization stage that exploits surrogate gradients while preserving the black-box status of HuBERT.

Targeted example:

```bash
python pgd_attack.py \
  --pseudo-labels artifacts/pseudo_labels.json \
  --surrogate-model checkpoints/best_model \
  --sample-index 0 \
  --targeted \
  --target-phrase "OPEN THE DOOR" \
  --epsilon 0.002 \
  --alpha 0.0002 \
  --iterations 40 \
  --output-audio artifacts/adv.wav
```

Untargeted example:

```bash
python pgd_attack.py \
  --pseudo-labels artifacts/pseudo_labels.json \
  --surrogate-model checkpoints/best_model \
  --sample-index 0 \
  --epsilon 0.002 \
  --alpha 0.0002 \
  --iterations 40 \
  --output-audio artifacts/adv.wav
```

## Step 5: Transfer to the black-box target

The transfer evaluation stage re-queries HuBERT on clean and adversarial audio, then computes transfer success.

```bash
python transfer_eval.py \
  --pseudo-labels artifacts/pseudo_labels.json \
  --adversarial-audio artifacts/adv.wav \
  --sample-index 0 \
  --targeted \
  --target-phrase "OPEN THE DOOR" \
  --relaxed-substring-match
```

## Metrics

The pipeline is designed to surface paper-style metrics:

- `attack_cost.json`: total query count, attack generation time, average optimization iterations, and GPU memory usage when available.
- `results_metrics.json` and `results_metrics.csv`: sample-level transcripts, CER/WER, TASR, UASR, attack time, and query count.
- `surrogate_eval.json` and `surrogate_eval.csv`: surrogate imitation quality against HuBERT pseudo-labels.

Metric definitions:

- `Clean CER`: CER between clean HuBERT transcription and ground-truth transcript.
- `TASR`: successful targeted attacks divided by total samples. Success can use exact matching or relaxed substring matching.
- `UASR`: successful untargeted attacks divided by total samples, with default thresholds `CER >= 0.5` or `WER >= 0.5`.

## Query-efficiency experiments

Prepare a CSV with one row per query budget and columns:

```text
queries,transfer_success,surrogate_wer,tasr,uasr,clean_cer,avg_attack_time
```

Then run:

```bash
python scripts/run_query_efficiency.py --input-csv results/query_budget_summary.csv
python scripts/export_paper_tables.py --input-csv results/query_budget_summary.csv
```

This produces:

- summary CSV/JSON
- Matplotlib plots
- paper-ready Markdown and CSV tables

## Commercial API follow-up

To address reviewer concerns about unnamed commercial ASR endpoints, this
repository now includes a lightweight manifest-based workflow for external API
evaluation.

- `configs/commercial_api_targets.yaml`
  separates `Commercial black-box APIs` from `Modern foundation ASR APIs`
- `prepare_commercial_api_manifest.py`
  expands an existing attack CSV into clean/adversarial query rows
- `aggregate_commercial_api_results.py`
  converts filled transcripts into TASR/UASR/CER/WER summary rows
- `COMMERCIAL_API_EXPERIMENT.md`
  documents the recommended Google STT v2 `chirp_3` follow-up experiment

## Notes

- `train-clean-100` is expected for the main extraction experiment; if it is not present locally, provide the correct path.
- The code is modular enough to extend toward Common Voice, VoxCeleb, surrogate ensembles, spectrogram visualizations, and multiple target phrases.
- The current implementation favors research clarity and reproducibility over aggressive optimization. For large-scale runs on Palmetto, submit training and attack steps as Slurm jobs after reviewing resource needs.
