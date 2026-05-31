from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a surrogate Wav2Vec2 CTC model on pseudo-labels.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--pseudo-labels", default="artifacts/pseudo_labels.json")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import evaluate
    from transformers import AutoModelForCTC, AutoProcessor, Trainer, TrainingArguments

    from src.asr_blackbox.config import load_config
    from src.asr_blackbox.data import CTCCollator, load_pseudo_label_dataset, train_val_split
    from src.asr_blackbox.logging_utils import get_logger
    from src.asr_blackbox.metrics import normalize_text
    from src.asr_blackbox.models import load_audio

    config = load_config(args.config)
    logger = get_logger("train_surrogate")

    processor = AutoProcessor.from_pretrained(config["models"]["surrogate"])
    dataset = load_pseudo_label_dataset(args.pseudo_labels)
    splits = train_val_split(dataset, config["training"]["validation_ratio"], config["seed"])

    def prepare_batch(example: dict) -> dict:
        waveform, _ = load_audio(example["audio_path"])
        example["input_values"] = processor(waveform.numpy(), sampling_rate=16000).input_values[0]
        example["labels"] = processor.tokenizer(normalize_text(example["transcription"])).input_ids
        return example

    train_dataset = splits["train"].map(prepare_batch, remove_columns=splits["train"].column_names)
    eval_dataset = splits["test"].map(prepare_batch, remove_columns=splits["test"].column_names)

    model = AutoModelForCTC.from_pretrained(
        config["models"]["surrogate"],
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model.freeze_feature_encoder()

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions.argmax(axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_text = [normalize_text(text) for text in processor.batch_decode(pred_ids)]
        label_text = [normalize_text(text) for text in processor.batch_decode(pred.label_ids, group_tokens=False)]
        return {
            "wer": wer_metric.compute(predictions=pred_text, references=label_text),
            "cer": cer_metric.compute(predictions=pred_text, references=label_text),
        }

    output_dir = Path(config["paths"]["checkpoints"])
    logging_dir = Path(config["paths"]["logs"])
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=config["training"]["logging_steps"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["num_train_epochs"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        bf16=config["training"].get("bf16", False),
        fp16=config["training"]["fp16"],
        save_total_limit=config["training"]["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["tensorboard"],
        logging_dir=str(logging_dir),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=CTCCollator(processor),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(config["paths"]["best_model"])
    processor.save_pretrained(config["paths"]["best_model"])
    logger.info("Saved best surrogate model to %s", config["paths"]["best_model"])


if __name__ == "__main__":
    main()
