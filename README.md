# Qwen VL 4B QLoRA Fine-tuning

This repository contains source code and benchmark artifacts for QLoRA fine-tuning experiments on a Qwen VL 4B model.

## 1) Dataset and Labeling
- Data roots (from config):
  - `pdf_dir`: `/home/sehun/project/sehun/qwen_test/pdf`
  - `images_dir`: `/home/sehun/project/sehun/qwen_test/images`
  - `label_dir`: `/home/sehun/project/sehun/qwen_test/label`
- Teacher model for supervision: `qwen3.5-flash`
- Label generation setup:
  - `label_workers`: `32`
  - `label_max_retries`: `3`
  - `label_val_ratio`: `0.15`
  - `label_val_seed`: `42`
- Validation report is based on `100` held-out samples.

## 2) Training Method
- Base model: `Qwen/Qwen3.5-4B`
- Fine-tuning type: **QLoRA** (`rank=16`, `epochs=2`) 
- Prompting style for SFT: `flash_teacher`
- Inference/training infra (config snapshot):
  - backend: `vllm`
  - `vl_max_model_len=4096`
  - `vl_batch_size=8`
  - `vl_gpu_memory_utilization=0.45`

## 3) Evaluation Protocol
- Baseline: pre-SFT model inference with the same English prompt setting
- SFT: adapter-applied model outputs on the same validation split
- Teacher reference: `qwen3.5-flash` labels
- Main metrics: ROUGE-L F1, Token F1, BLEU-1, char similarity, hallucination heuristic

## 4) Key Benchmark Results
| Metric | Baseline | SFT | Gain |
|---|---:|---:|---:|
| ROUGE-L F1 | 0.225 | 0.260 | +0.035 |
| Token F1 | 0.357 | 0.413 | +0.055 |
| BLEU-1 | 0.399 | 0.462 | +0.063 |
| Hallucination heuristic | 32.0% | 17.0% | -15.0pp |

## 5) Repro Commands
```bash
# baseline
CUDA_VISIBLE_DEVICES=0 python scripts/run_baseline.py

# QLoRA SFT
CUDA_VISIBLE_DEVICES=0 python scripts/train_sft.py --config config_qwen35_4b.yaml --qlora --run-name sft_4b_qlora_r16_ep2

# evaluation
CUDA_VISIBLE_DEVICES=0 python scripts/eval_qwen35_4b.py
```

## 6) Repository Contents
- Training/eval source scripts in `scripts/`
- Experiment config in `config_qwen35_4b.yaml`
- Benchmark report in `docs/BASELINE_VS_SFT_REPORT.md`
- Metric summary in `docs/metrics_summary.json`
- Visualization in `assets/project1_qwen_qlora_metrics.png`
