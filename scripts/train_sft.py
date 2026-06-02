
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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


class SFTDataset(Dataset):
    """
    각 샘플: image + prompt → teacher_label
    - student: user vl_prompt only (default)
    - flash_teacher: teacher_system + metadata/draft user (qwen3.5-flash labeling)
    """

    def __init__(
        self,
        rows: List[dict],
        processor: AutoProcessor,
        base_dir: Path,
        max_len: int = 1024,
        *,
        prompt_style: str = "student",
        teacher_system: str = "",
        family: str = "qwen2_vl",
    ):
        self.rows = rows
        self.processor = processor
        self.base_dir = base_dir
        self.max_len = max_len
        self.prompt_style = prompt_style
        self.teacher_system = teacher_system
        self.family = family
        from vlm_common import build_flash_teacher_messages, chat_template_kwargs

        self._build_flash = build_flash_teacher_messages
        self._template_kwargs = chat_template_kwargs(family)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        img_path = self.base_dir / row["image_path"]
        image = Image.open(img_path).convert("RGB")

        target_text: str = row["teacher_label"]

        if self.prompt_style == "flash_teacher":
            messages_full = self._build_flash(
                row,
                image,
                system=self.teacher_system,
                include_assistant=True,
                target_text=target_text,
            )
            messages_prompt = self._build_flash(
                row, image, system=self.teacher_system, include_assistant=False
            )
        else:
            prompt_text: str = row["student_vl_prompt"]
            messages_full = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ],
                },
                {"role": "assistant", "content": target_text},
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
            messages_full,
            tokenize=False,
            add_generation_prompt=False,
            **self._template_kwargs,
        )
        inputs_full = self.processor(
            text=[text_full],
            images=[image],
            return_tensors="pt",
        )

        # prompt-only 시퀀스 (마스킹 길이 계산용)
        text_prompt = self.processor.apply_chat_template(
            messages_prompt,
            tokenize=False,
            add_generation_prompt=True,
            **self._template_kwargs,
        )
        inputs_prompt = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
        )

        input_ids = inputs_full["input_ids"][0]
        attention_mask = inputs_full["attention_mask"][0]
        prompt_len = inputs_prompt["input_ids"].shape[1]

        # max_len 초과 시 뒤쪽 텍스트 부분만 잘라냄
        if input_ids.shape[0] > self.max_len:
            input_ids = input_ids[: self.max_len]
            attention_mask = attention_mask[: self.max_len]

        labels = input_ids.clone()
        # prompt 토큰 마스킹 → loss 계산 제외
        labels[:prompt_len] = -100

        out: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # 비전 토큰 (pixel_values, image_grid_thw 등)
        # pixel_values: [num_patches, C, H, W]  — batch dim 없음, 그대로 유지
        # image_grid_thw: [num_images, 3]        — 마찬가지
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
    pad_extra_keys: frozenset[str] = frozenset({"mm_token_type_ids"})

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
                v = f[k]
                if k in self.pad_extra_keys and hasattr(v, "shape") and v.dim() >= 1:
                    seq = v[0] if v.dim() > 1 and v.shape[0] == 1 else v
                    if seq.dim() == 1:
                        extra[k].append(
                            torch.cat([seq, torch.zeros(pad, dtype=seq.dtype)])
                        )
                        continue
                extra[k].append(v)

        batch: Dict[str, torch.Tensor] = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
        }
        for k, vs in extra.items():
            if k in self.cat_keys:
                # pixel_values: [patches, C, H, W] → cat across batch
                # image_grid_thw: [1, 3] per sample → cat → [B, 3]
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
    p = argparse.ArgumentParser()
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
        help="override config label_dir (e.g. label_ko for Korean SFT)",
    )
    return p.parse_args()


def main() -> None:
    # 모델 로드 전에 GPU 설정 — multi-GPU DataParallel은 VLM patch collation과 충돌
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    args = parse_args()
    cfg = _load_config(args.config)

    # FP8 모델은 훈련 불가 → sft_model_id(bf16) 우선 사용
    model_id: str = cfg.get("sft_model_id", cfg.get("vl_model_id", "Qwen/Qwen2-VL-2B-Instruct"))
    from vlm_common import collator_cat_keys, load_processor, model_family

    family = model_family(model_id)

    def _resolve_dir(key: str, default: Path) -> Path:
        """config 경로가 실제로 존재하면 사용, 아니면 QWEN_TEST 기준 fallback."""
        p = Path(cfg.get(key, ""))
        if p and p.exists():
            return p
        return default

    if args.label_dir is not None:
        label_dir = args.label_dir
    else:
        label_dir = _resolve_dir("label_dir", QWEN_TEST / "label")
    output_dir = _resolve_dir("output_dir", QWEN_TEST / "output")

    mode = "qlora" if args.qlora else "lora"
    run_name = args.run_name or f"sft_{mode}_r{args.lora_r}_ep{args.epochs}"
    sft_out = output_dir / "sft" / run_name
    sft_out.mkdir(parents=True, exist_ok=True)

    prompt_style = str(cfg.get("sft_prompt_style", "student")).strip()
    teacher_system = str(cfg.get("teacher_system", "")).strip()
    if prompt_style == "flash_teacher" and not teacher_system:
        raise ValueError("sft_prompt_style=flash_teacher requires teacher_system in config")

    print(f"▶  model : {model_id}")
    print(f"▶  mode  : {mode.upper()}")
    print(f"▶  prompt: {prompt_style}")
    print(f"▶  run   : {run_name}")
    print(f"▶  output: {sft_out}")

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    train_rows = _read_jsonl(label_dir / "train.jsonl")
    val_rows = _read_jsonl(label_dir / "val.jsonl")

    # quality_ok 필터 (teacher_label 없는 샘플 제거)
    train_rows = [r for r in train_rows if r.get("teacher_label")]
    val_rows = [r for r in val_rows if r.get("teacher_label")]

    if args.dry_run:
        train_rows = train_rows[:10]
        val_rows = val_rows[:5]
        print(f"⚡ dry-run: train={len(train_rows)} val={len(val_rows)}")

    print(f"📊 train={len(train_rows)} val={len(val_rows)}")

    # ── Processor & Model ────────────────────────────────────────────────────
    # max_pixels: 이미지당 최대 패치 수 제한 → 시퀀스 길이 예측 가능하게 유지
    # 256*28*28 ≈ 200k pixels → 약 256 visual tokens per image
    processor = load_processor(model_id, max_pixels=256 * 28 * 28)
    model = _load_model(
        model_id,
        qlora=args.qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    # ── Dataset & Collator ───────────────────────────────────────────────────
    ds_kw = dict(
        prompt_style=prompt_style,
        teacher_system=teacher_system,
        family=family,
    )
    train_ds = SFTDataset(train_rows, processor, QWEN_TEST, max_len=args.max_len, **ds_kw)
    val_ds = SFTDataset(val_rows, processor, QWEN_TEST, max_len=args.max_len, **ds_kw)
    collator = SFTCollator(
        processor=processor,
        pad_token_id=pad_id,
        cat_keys=collator_cat_keys(family),
    )

    # ── Training Arguments ───────────────────────────────────────────────────
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

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("🚀 Training start...")
    trainer.train()

    # ── Save LoRA adapter ────────────────────────────────────────────────────
    adapter_path = sft_out / "adapter_final"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    print(f"✅ Adapter saved → {adapter_path}")


if __name__ == "__main__":
    main()
