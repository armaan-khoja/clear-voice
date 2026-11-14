"""Fine-tuning Whisper on TORGO clips with Hugging Face's Seq2SeqTrainer.

Follows the recipe in Hugging Face's "Fine-Tune Whisper" blog: log-mel features are computed per batch,
labels are the transcript's tokens, and validation WER chooses the checkpoint to keep.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from .metrics import score


@dataclass
class TrainConfig:
    base_model: str = "openai/whisper-small"
    learning_rate: float = 1e-5
    batch_size: int = 16
    grad_accum: int = 1
    warmup_frac: float = 0.1
    max_steps: int | None = 2000         # used unless None
    num_train_epochs: float | None = None  # used when max_steps is None
    eval: bool = True
    eval_steps: int = 250
    save_steps: int = 250
    patience: int = 3                    # evaluations without improvement before stopping
    eval_batch_size: int = 16
    generation_max_length: int = 128
    gradient_checkpointing: bool = False
    logging_steps: int = 25
    num_workers: int = 2
    seed: int = 42


class _Clips(torch.utils.data.Dataset):
    """Each item: the raw 16 kHz audio and the transcript's token ids."""

    def __init__(self, rows: pd.DataFrame, audio: dict, processor):
        self.audio = [audio[int(r)] for r in rows["row_id"]]
        self.labels = [processor.tokenizer(t).input_ids for t in rows["text"]]

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, i):
        return {"audio": self.audio[i], "labels": self.labels[i]}


class _Collator:
    """Turns a batch of raw clips into log-mel features and padded labels (-100 = ignored)."""

    def __init__(self, processor, decoder_start_token_id: int):
        self.processor, self.start = processor, decoder_start_token_id

    def __call__(self, items):
        feats = self.processor.feature_extractor(
            [np.asarray(it["audio"], dtype=np.float32) for it in items], sampling_rate=16000, return_tensors="pt"
        ).input_features
        labels = self.processor.tokenizer.pad({"input_ids": [it["labels"] for it in items]}, return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        if (ids[:, 0] == self.start).all():  # the model adds the start token itself
            ids = ids[:, 1:]
        return {"input_features": feats, "labels": ids}


def finetune(cfg: TrainConfig, train_rows: pd.DataFrame, val_rows: pd.DataFrame | None,
             audio: dict, output_dir: str) -> str:
    """Fine-tune `cfg.base_model` and return the folder holding the kept model and its processor."""
    out = Path(output_dir)
    processor = WhisperProcessor.from_pretrained(cfg.base_model)
    processor.tokenizer.set_prefix_tokens(language="english", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(cfg.base_model)  # float32 master weights
    model.generation_config.language, model.generation_config.task = "en", "transcribe"
    model.generation_config.forced_decoder_ids = None
    if cfg.gradient_checkpointing:
        model.config.use_cache = False

    use_eval = cfg.eval and val_rows is not None and len(val_rows) > 0
    cuda = torch.cuda.is_available()
    steps = cfg.max_steps if cfg.max_steps is not None else -1
    total = steps if steps > 0 else int(np.ceil(len(train_rows) / (cfg.batch_size * cfg.grad_accum))
                                        * (cfg.num_train_epochs or 1))
    args = Seq2SeqTrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_steps=max(1, int(cfg.warmup_frac * total)),
        max_steps=steps,
        num_train_epochs=cfg.num_train_epochs or 1,
        eval_strategy="steps" if use_eval else "no",
        eval_steps=cfg.eval_steps,
        save_strategy="steps" if use_eval else "no",
        save_steps=cfg.save_steps,
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=use_eval,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=cfg.generation_max_length,
        bf16=cuda and torch.cuda.is_bf16_supported(),
        use_cpu=not cuda,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        dataloader_num_workers=cfg.num_workers,
        remove_unused_columns=False,
        report_to="none",
        seed=cfg.seed,
    )

    def compute_metrics(pred):
        label_ids = np.where(pred.label_ids == -100, processor.tokenizer.pad_token_id, pred.label_ids)
        hyps = processor.batch_decode(pred.predictions, skip_special_tokens=True)
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        s = score(refs, hyps)
        return {"wer": s["wer"], "cer": s["cer"]}

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=_Clips(train_rows, audio, processor),
        eval_dataset=_Clips(val_rows, audio, processor) if use_eval else None,
        data_collator=_Collator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics if use_eval else None,
        processing_class=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience)] if use_eval else None,
    )
    trainer.train()
    final = out / "best"
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    return str(final)
