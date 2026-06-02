# Qwen3.5-4B SFT — Validation quantitative report

- **Samples:** 100 (val, seed=42)
- **Base:** Qwen/Qwen3.5-4B
- **Adapter:** /home/sehun/project/sehun/qwen_test/output/qwen35_4b/sft/sft_4b_qlora_r16_ep2/adapter_final
- **Baseline:** live pre-SFT inference (same English prompt as SFT)
- **Teacher:** qwen3.5-flash (`label/`)

## Similarity vs teacher (higher = better)

| Metric | Baseline | SFT | Δ mean | SFT win % |
|--------|----------|-----|--------|-----------|
| ROUGE-L F1 | 0.225 | 0.260 | +0.035 | 69.0% |
| Token F1 | 0.357 | 0.413 | +0.055 | 80.0% |
| BLEU-1 | 0.399 | 0.462 | +0.063 | 80.0% |
| Char similarity | 0.234 | 0.246 | +0.013 | 64.0% |

## Abnormal / hallucination rates (validation)

| Metric | Baseline | After SFT | Δ |
|--------|----------|-----------|---|
| Any abnormal response | 100.0% | 100.0% | +0.0pp |
| Hallucination heuristic | 32.0% | 17.0% | -15.0pp |
| Samples fixed (issue→clean) | — | 0 | — |

## Format compliance

| Rule | Baseline | SFT |
|------|----------|-----|
| Forbidden prefix | 0.0% | 0.0% |
| Exactly 2 sentences | 0.0% | 0.0% |
| Mean words | 83.0 | 82.9 |

Artifacts: `/home/sehun/project/sehun/qwen_test/output/qwen35_4b/sft/sft_4b_qlora_r16_ep2/metrics/`
