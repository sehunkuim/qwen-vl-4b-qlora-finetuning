# Qwen VL 4B QLoRA Fine-tuning

This repository contains source code and benchmark artifacts for QLoRA fine-tuning experiments on a Qwen VL 4B model.

## Key Gains
- ROUGE-L F1: `0.225 -> 0.260` (`+0.035`)
- Token F1: `0.357 -> 0.413` (`+0.055`)
- BLEU-1: `0.399 -> 0.462` (`+0.063`)
- Hallucination heuristic: `32% -> 17%` (`-15pp`)

## Included Files
- Training/eval source scripts in `scripts/`
- Experiment config in `config_qwen35_4b.yaml`
- Benchmark report in `docs/BASELINE_VS_SFT_REPORT.md`
- Metric summary in `docs/metrics_summary.json`
- Visualization in `assets/project1_qwen_qlora_metrics.png`
