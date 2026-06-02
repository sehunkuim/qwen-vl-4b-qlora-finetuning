"""
한국어 VLM 캡션 LoRA / QLoRA SFT (Qwen2-VL-2B)

- 데이터: label_ko/train.jsonl, val.jsonl (한국어 teacher_label)
- 입력 프롬프트: config vl_prompt_ko 또는 jsonl student_vl_prompt
- 실행 예:
  python scripts/train_sft\\ copy.py
  python scripts/train_sft\\ copy.py --qlora --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
)

QWEN_TEST = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_dir(cfg: dict, key: str, default: Path) -> Path:
    raw = cfg.get(key, "")
    if raw:
        return Path(raw)
    return default


def _student_prompt(row: dict, cfg: dict) -> str:
    """jsonl student_vl_prompt 우선, 없으면 config vl_prompt_ko."""
    if row.get("student_vl_prompt"):
        return row["student_vl_prompt"]
    return (cfg.get("vl_prompt_ko") or "").strip()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _filter_ko_rows(rows: List[dict]) -> List[dict]:
    """한국어 SFT용: teacher_label 필수, quality_ok 권장, lang=ko 우선."""
    out = []
    for r in rows:
        label = (r.get("teacher_label") or "").strip()
        if not label:
            continue
        if r.get("quality_ok") is False:
            continue
        out.append(r)
    return out


class KoreanSFTDataset(Dataset):
    """
    각 샘플: image + 한국어 student 프롬프트 → 한국어 teacher_label
    processor.apply_chat_template 로 대화를 포맷하고,
    labels 에서 prompt 부분을 -100 으로 마스킹.
    """

    def __init__(
        self,
        rows: List[dict],
        processor: AutoProcessor,
        base_dir: Path,
        cfg: dict,
        max_len: int = 2048,
    ):
        self.rows = rows
        self.processor = processor
        self.base_dir = base_dir
        self.cfg = cfg
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        img_path = self.base_dir / row["image_path"]
        image = Image.open(img_path).convert("RGB")

        prompt_text = _student_prompt(row, self.cfg)
        target_text: str = row["teacher_label"]

        messages_full = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": target_text,
            },
        ]

        messages_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        text_full = self.processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )
        inputs_full = self.processor(
            text=[text_full],
            images=[image],
            return_tensors="pt",
        )

        text_prompt = self.processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        inputs_prompt = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
        )

        input_ids = inputs_full["input_ids"][0]
        attention_mask = inputs_full["attention_mask"][0]
        prompt_len = inputs_prompt["input_ids"].shape[1]

        if input_ids.shape[0] > self.max_len:
            input_ids = input_ids[: self.max_len]
            attention_mask = attention_mask[: self.max_len]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        out: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        for k in inputs_full:
            if k not in ("input_ids", "attention_mask"):
                out[k] = inputs_full[k]

        return out


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

@dataclass
class SFTCollator:
    processor: Any
    pad_token_id: int
    cat_keys: frozenset[str] = frozenset({"pixel_values", "image_grid_thw"})

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].shape[0] for f in features)

        input_ids_list, attention_mask_list, labels_list = [], [], []
        extra_keys = [k for k in features[0] if k not in ("input_ids", "attention_mask", "labels")]
        extra: Dict[str, list] = {k: [] for k in extra_keys}

        for f in features:
            seq_len = f["input_ids"].shape[0]
            pad = max_len - seq_len

            input_ids_list.append(
                torch.cat([f["input_ids"], torch.full((pad,), self.pad_token_id)])
            )
            attention_mask_list.append(
                torch.cat([f["attention_mask"], torch.zeros(pad, dtype=torch.long)])
            )
            labels_list.append(
                torch.cat([f["labels"], torch.full((pad,), -100)])
            )
            for k in extra_keys:
                extra[k].append(f[k])

        batch: Dict[str, torch.Tensor] = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
        }
        for k, vs in extra.items():
            if k in self.cat_keys:
                try:
                    batch[k] = torch.cat(vs, dim=0)
                except Exception:
                    batch[k] = vs
            else:
                try:
                    batch[k] = torch.stack(vs)
                except Exception:
                    batch[k] = vs

        return batch


# ---------------------------------------------------------------------------
# Model loading (LoRA / QLoRA)
# ---------------------------------------------------------------------------

def _load_model(model_id: str, qlora: bool, lora_r: int, lora_alpha: int, lora_dropout: float):
    from vlm_common import load_sft_model

    return load_sft_model(
        model_id,
        qlora=qlora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="한국어 강의 슬라이드 캡션 LoRA/QLoRA SFT (Qwen2-VL)",
    )
    p.add_argument("--config", type=Path, default=QWEN_TEST / "config.yaml")
    p.add_argument("--qlora", action="store_true", help="4-bit NF4 QLoRA")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2, help="per-device train batch")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--dry-run", action="store_true", help="10샘플 smoke test")
    p.add_argument(
        "--label-dir",
        type=Path,
        default=None,
        help="한국어 라벨 디렉터리 (기본: config label_ko_dir 또는 label_ko)",
    )
    return p.parse_args()


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    args = parse_args()
    cfg = _load_config(args.config)

    model_id: str = cfg.get("sft_model_id", cfg.get("vl_model_id", "Qwen/Qwen2-VL-2B-Instruct"))
    from vlm_common import collator_cat_keys, load_processor, model_family

    family = model_family(model_id)

    if args.label_dir is not None:
        label_dir = args.label_dir
    else:
        label_dir = _resolve_dir(cfg, "label_ko_dir", QWEN_TEST / "label_ko")

    output_dir = _resolve_dir(cfg, "output_dir", QWEN_TEST / "output")

    mode = "qlora" if args.qlora else "lora"
    run_name = args.run_name or f"sft_ko_{mode}_r{args.lora_r}_ep{args.epochs}"
    sft_out = output_dir / "sft" / run_name
    sft_out.mkdir(parents=True, exist_ok=True)

    print(f"▶  lang  : ko")
    print(f"▶  model : {model_id}")
    print(f"▶  mode  : {mode.upper()}")
    print(f"▶  labels: {label_dir}")
    print(f"▶  run   : {run_name}")
    print(f"▶  output: {sft_out}")

    train_rows = _filter_ko_rows(_read_jsonl(label_dir / "train.jsonl"))
    val_rows = _filter_ko_rows(_read_jsonl(label_dir / "val.jsonl"))

    if args.dry_run:
        train_rows = train_rows[:10]
        val_rows = val_rows[:5]
        print(f"⚡ dry-run: train={len(train_rows)} val={len(val_rows)}")

    if not train_rows:
        raise SystemExit(
            f"학습 샘플이 없습니다. {label_dir}/train.jsonl 을 확인하거나 "
            "make_teacher_labels.py --lang ko 로 라벨을 생성하세요."
        )

    print(f"📊 train={len(train_rows)} val={len(val_rows)}")

    processor = load_processor(model_id, max_pixels=256 * 28 * 28)
    model = _load_model(
        model_id,
        qlora=args.qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    train_ds = KoreanSFTDataset(train_rows, processor, QWEN_TEST, cfg, max_len=args.max_len)
    val_ds = KoreanSFTDataset(val_rows, processor, QWEN_TEST, cfg, max_len=args.max_len)
    collator = SFTCollator(
        processor=processor,
        pad_token_id=pad_id,
        cat_keys=collator_cat_keys(family),
    )

    training_args = TrainingArguments(
        output_dir=str(sft_out),
        run_name=run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        fp16=False,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("🚀 한국어 LoRA 학습 시작...")
    trainer.train()

    adapter_path = sft_out / "adapter_final"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    print(f"✅ 한국어 adapter 저장 → {adapter_path}")


if __name__ == "__main__":
    main()
