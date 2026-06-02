#!/usr/bin/env python3
"""
Qwen3.5-4B: val inference (base + SFT) and quantitative report vs teacher.

Usage:
  python scripts/eval_qwen35_4b.py --adapter output/qwen35_4b/sft/sft_4b_qlora_r16_ep2/adapter_final
  python scripts/eval_qwen35_4b.py --skip-inference --captions output/qwen35_4b/sft/.../sft_captions_100.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

QWEN_TEST = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QWEN_TEST / "scripts"))


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _select_val(rows: list, limit: int, seed: int) -> list:
    val = [r for r in rows if r.get("split") == "val" and r.get("teacher_label")]
    rng = random.Random(seed)
    if len(val) > limit:
        val = rng.sample(val, limit)
    return sorted(val, key=lambda r: r.get("id", ""))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=QWEN_TEST / "config_qwen35_4b.yaml")
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--captions", type=Path, default=None)
    p.add_argument("--infer-baseline-only", action="store_true")
    args = p.parse_args()

    cfg = _load_config(args.config)
    base_id = str(cfg.get("sft_model_id", "Qwen/Qwen3.5-4B"))
    label_dir = Path(cfg.get("label_dir", QWEN_TEST / "label"))
    out_root = Path(cfg.get("output_dir", QWEN_TEST / "output/qwen35_4b"))

    adapter = args.adapter
    if adapter is None:
        adapter = out_root / "sft" / "sft_4b_qlora_teacher_r16_ep2" / "adapter_final"
    adapter = adapter.resolve()

    run_dir = adapter.parent
    captions_path = args.captions or (run_dir / f"sft_captions_{args.limit}.json")

    if not args.skip_inference:
        from PIL import Image
        from run_sft_compare import _caption_one, _load_model, _read_jsonl

        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        rows = _read_jsonl(label_dir / "records.jsonl")
        samples = _select_val(rows, args.limit, args.seed)
        prompt_style = str(cfg.get("sft_prompt_style", "student")).strip()
        teacher_system = str(cfg.get("teacher_system", "")).strip()
        prompt = str(cfg.get("vl_prompt", "")).strip()
        cap_kw = dict(
            prompt_style=prompt_style,
            teacher_system=teacher_system,
        )
        print(f"📊 val samples={len(samples)} base={base_id} prompt={prompt_style}")

        import torch

        t0 = time.perf_counter()
        results = [{**row, "baseline_caption": "", "sft_caption": ""} for row in samples]

        print("🔄 Base model (pre-SFT) captions...")
        base_model, proc = _load_model(base_id, None, "0")
        for i, row in enumerate(results, 1):
            img = Image.open(QWEN_TEST / row["image_path"]).convert("RGB")
            row["baseline_caption"] = _caption_one(
                base_model, proc, img, prompt, 180, row=row, **cap_kw
            )
            if i % 5 == 0 or i == len(results):
                print(f"  baseline [{i}/{len(results)}]", flush=True)
        del base_model
        torch.cuda.empty_cache()

        if not args.infer_baseline_only:
            print("🔄 SFT (LoRA) captions...")
            sft_model, proc2 = _load_model(base_id, adapter, "0")
            for i, row in enumerate(results, 1):
                img = Image.open(QWEN_TEST / row["image_path"]).convert("RGB")
                row["sft_caption"] = _caption_one(
                    sft_model, proc2, img, prompt, 180, row=row, **cap_kw
                )
                if i % 5 == 0 or i == len(results):
                    print(f"  sft [{i}/{len(results)}]", flush=True)
            del sft_model
            torch.cuda.empty_cache()

        meta = {
            "phase": "qwen35_4b_val_eval",
            "base_model_id": base_id,
            "adapter_path": str(adapter) if not args.infer_baseline_only else None,
            "baseline_source": "live Qwen3.5-4B inference (no adapter)",
            "prompt_style": prompt_style if not args.skip_inference else None,
            "vl_prompt": prompt if prompt_style == "student" else "flash_teacher",
            "sample_count": len(results),
            "sample_seed": args.seed,
            "split": "val",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "caption_seconds": round(time.perf_counter() - t0, 3),
        }
        with captions_path.open("w", encoding="utf-8") as f:
            json.dump({"meta": meta, "samples": results}, f, ensure_ascii=False, indent=2)
        print(f"📁 {captions_path}")

    if not captions_path.is_file():
        raise FileNotFoundError(captions_path)

    metrics_dir = run_dir / "metrics"
    subprocess.run(
        [
            sys.executable,
            str(QWEN_TEST / "scripts/plot_sft_metrics.py"),
            "--captions",
            str(captions_path),
            "--out-dir",
            str(metrics_dir),
        ],
        check=True,
        cwd=str(QWEN_TEST),
    )
    subprocess.run(
        [
            sys.executable,
            str(QWEN_TEST / "scripts/plot_issue_tables_en.py"),
            "--captions",
            str(captions_path),
            "--out-dir",
            str(metrics_dir / "issue_tables"),
        ],
        check=True,
        cwd=str(QWEN_TEST),
    )

    with captions_path.open(encoding="utf-8") as f:
        data = json.load(f)
    with (metrics_dir / "metrics_summary.json").open(encoding="utf-8") as f:
        metrics = json.load(f)
    with (metrics_dir / "issue_tables" / "issue_summary.json").open(encoding="utf-8") as f:
        issues = json.load(f)

    sim = metrics["similarity_vs_teacher"]
    report = f"""# Qwen3.5-4B SFT — Validation quantitative report

- **Samples:** {data['meta'].get('sample_count', len(data['samples']))} (val, seed={args.seed})
- **Base:** {base_id}
- **Adapter:** {adapter}
- **Baseline:** live pre-SFT inference (same English prompt as SFT)
- **Teacher:** qwen3.5-flash (`label/`)

## Similarity vs teacher (higher = better)

| Metric | Baseline | SFT | Δ mean | SFT win % |
|--------|----------|-----|--------|-----------|
"""
    for m in sim.values():
        d = m["sft_mean"] - m["baseline_mean"]
        report += (
            f"| {m['label']} | {m['baseline_mean']:.3f} | {m['sft_mean']:.3f} "
            f"| {d:+.3f} | {m['sft_win_rate']*100:.1f}% |\n"
        )

    fc = metrics["format_compliance"]
    report += f"""
## Abnormal / hallucination rates (validation)

| Metric | Baseline | After SFT | Δ |
|--------|----------|-----------|---|
| Any abnormal response | {issues['any_issue_baseline_pct']:.1f}% | {issues['any_issue_sft_pct']:.1f}% | {issues['any_issue_sft_pct'] - issues['any_issue_baseline_pct']:+.1f}pp |
| Hallucination heuristic | {issues['hallucination_baseline_pct']:.1f}% | {issues['hallucination_sft_pct']:.1f}% | {issues['hallucination_sft_pct'] - issues['hallucination_baseline_pct']:+.1f}pp |
| Samples fixed (issue→clean) | — | {issues['fixed_any_issue_count']} | — |

## Format compliance

| Rule | Baseline | SFT |
|------|----------|-----|
| Forbidden prefix | {fc['forbidden_prefix_rate']['baseline']*100:.1f}% | {fc['forbidden_prefix_rate']['sft']*100:.1f}% |
| Exactly 2 sentences | {fc['exact_two_sentences_rate']['baseline']*100:.1f}% | {fc['exact_two_sentences_rate']['sft']*100:.1f}% |
| Mean words | {fc['mean_word_count']['baseline']:.1f} | {fc['mean_word_count']['sft']:.1f} |

Artifacts: `{metrics_dir}/`
"""
    report_path = run_dir / "BASELINE_VS_SFT_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"📄 {report_path}")


if __name__ == "__main__":
    main()
