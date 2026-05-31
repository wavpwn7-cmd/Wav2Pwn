# Commercial API Follow-Up Experiment

This protocol addresses the reviewer comment that the paper's `Google Cloud STT`,
`Azure`, `Amazon Transcribe`, and `IBM Watson` rows are not reproducible enough
when the exact API versions and model identifiers are omitted.

## Goal

Split the old bucket into two explicit groups:

1. `Commercial black-box APIs`
2. `Modern foundation ASR APIs`

The first group preserves the paper's original rows for reproduction. The second
group tests newer named API endpoints, starting with Google Speech-to-Text v2
`chirp_3`.

## Why this helps

- It removes the incorrect implication that the commercial rows are
  definitively `non-SSL`.
- It separates "paper reproduction with incomplete provider metadata" from
  "modern named API evaluation".
- It makes the Google v2 / Chirp follow-up experiment directly reportable in
  the same Table IV format.

## Files

- `configs/commercial_api_targets.yaml`
  defines the table groups and target metadata
- `prepare_commercial_api_manifest.py`
  expands an existing attack CSV into a query manifest for external APIs
- `aggregate_commercial_api_results.py`
  converts completed transcripts back into TASR/UASR/CER/WER table rows

## Recommended first experiment

Use the same adversarial audio already generated for the paper pipeline and add:

- `Google Cloud STT v2 (chirp_3)` under `Modern foundation ASR APIs`

That gives a clean answer to the reviewer:

- If `chirp_3` behaves differently from the paper's `Google Cloud STT` row,
  then the missing model/version detail materially affects the claim.
- If it behaves similarly, the paper's conclusion becomes stronger because it
  generalizes to a named modern API.

## Workflow

1. Generate or reuse attack CSVs such as `results/wav2pwn_10.csv`.
2. Build a commercial API query manifest:

```bash
cd blackbox_surrogate_asr
python prepare_commercial_api_manifest.py \
  --attack-csv results/wav2pwn_10.csv \
  --targets-config configs/commercial_api_targets.yaml \
  --output-csv results/commercial_api_query_manifest.csv
```

3. For each row, query the listed API on `audio_path` and write the returned
   transcript into `api_transcript`.
4. Aggregate results back into Table IV format:

```bash
python aggregate_commercial_api_results.py \
  --manifest-csv results/commercial_api_query_manifest.csv \
  --output-csv results/commercial_api_table4.csv \
  --output-md results/commercial_api_table4.md \
  --relaxed-substring-match
```

## Manifest format

Each target/sample pair expands into two rows:

- `audio_variant=clean`
- `audio_variant=adversarial`

The manifest keeps:

- target metadata: group, provider, API version, model name
- attack metadata: sample id, target phrase
- transcript slot: `api_transcript`

This makes it easy to batch-submit requests externally and still aggregate them
with exact model/version provenance.
