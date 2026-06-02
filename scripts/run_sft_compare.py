#!/usr/bin/env python3
"""
Caption images with SFT LoRA adapter and build an HTML comparison vs baseline.

Usage:
  python scripts/run_sft_compare.py
  python scripts/run_sft_compare.py --limit 100 --adapter output/sft/sft_lora_r16_ep2/adapter_final
  python scripts/run_sft_compare.py --dry-run 3   # quick smoke test

Outputs:
  output/sft/<run>/sft_captions_<N>.json
  output/sft/<run>/compare_baseline_vs_sft_<N>.html
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote

import yaml
from PIL import Image

QWEN_TEST = Path(__file__).resolve().parent.parent


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _select_samples(rows: List[dict], limit: int, seed: int) -> List[dict]:
    """Prefer val split; stable shuffle for reproducibility."""
    val_rows = [r for r in rows if r.get("split") == "val" and r.get("baseline_caption")]
    pool = val_rows if len(val_rows) >= limit else [r for r in rows if r.get("baseline_caption")]
    rng = random.Random(seed)
    if len(pool) > limit:
        pool = rng.sample(pool, limit)
    else:
        pool = pool[:limit]
    return sorted(pool, key=lambda r: r.get("id", ""))


def _load_model(base_id: str, adapter_dir: Path | None, device: str):
    import torch
    from peft import PeftModel
    from vlm_common import load_processor, load_sft_model, model_family

    family = model_family(base_id)
    if adapter_dir is not None:
        processor = load_processor(str(adapter_dir), max_pixels=256 * 28 * 28)
    else:
        processor = load_processor(base_id, max_pixels=256 * 28 * 28)

    if family == "qwen3_5":
        from transformers import AutoModelForImageTextToText

        model_cls = AutoModelForImageTextToText
    else:
        from transformers import AutoModelForVision2Seq

        model_cls = AutoModelForVision2Seq

    base = model_cls.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        device_map={"": int(device)},
        trust_remote_code=True,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(base, str(adapter_dir))
    else:
        model = base
    model.eval()
    return model, processor


def _caption_one(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    *,
    row: dict | None = None,
    prompt_style: str = "student",
    teacher_system: str = "",
) -> str:
    import torch
    from vlm_common import build_flash_teacher_messages, chat_template_kwargs, model_family

    if prompt_style == "flash_teacher" and row is not None:
        messages = build_flash_teacher_messages(
            row, image, system=teacher_system, include_assistant=False
        )
        family = model_family(
            getattr(model, "name_or_path", None)
            or getattr(getattr(model, "base_model", None), "name_or_path", "Qwen/Qwen3.5-4B")
        )
        tmpl_kw = chat_template_kwargs(family)
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        tmpl_kw = chat_template_kwargs(model_family(getattr(model, "name_or_path", "")))

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **tmpl_kw
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out_ids[0, inputs["input_ids"].shape[1] :]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


def _image_src(
    image_path: str,
    out_html: Path,
    base_dir: Path,
    *,
    embed: bool,
    max_width: int,
) -> str:
    """Return img src: base64 data URI (file:// safe) or URL-encoded relative path."""
    full = base_dir / image_path
    if embed:
        im = Image.open(full).convert("RGB")
        w, h = im.size
        if w > max_width:
            im = im.resize((max_width, int(h * max_width / w)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    try:
        rel = Path(os.path.relpath(base_dir, out_html.parent)) / image_path
    except ValueError:
        rel = Path(image_path)
    return "/".join(quote(part, safe="") for part in rel.as_posix().split("/"))


def _run_captioning(
    samples: List[dict],
    model,
    processor,
    prompt: str,
    base_dir: Path,
    max_new_tokens: int,
) -> List[dict]:
    results = []
    t0 = time.perf_counter()
    for i, row in enumerate(samples, 1):
        img_path = base_dir / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        cap = _caption_one(model, processor, image, prompt, max_new_tokens)
        results.append(
            {
                **{k: row[k] for k in ("id", "folder", "file", "image_path", "page", "pdf_name")},
                "baseline_caption": row.get("baseline_caption", ""),
                "teacher_label": row.get("teacher_label", ""),
                "sft_caption": cap,
            }
        )
        if i % 10 == 0 or i == len(samples):
            elapsed = time.perf_counter() - t0
            print(f"  [{i}/{len(samples)}] {elapsed:.1f}s ({elapsed / i:.2f}s/img)")
    return results


def _build_html(
    results: List[dict],
    out_html: Path,
    meta: Dict[str, Any],
    *,
    base_dir: Path = QWEN_TEST,
    embed_images: bool = True,
    thumb_max_width: int = 480,
) -> None:
    """Build comparison HTML. Default: embed JPEG thumbnails (works with file://)."""
    cards = []
    for i, r in enumerate(results, 1):
        src = _image_src(
            r["image_path"],
            out_html,
            base_dir,
            embed=embed_images,
            max_width=thumb_max_width,
        )
        if i % 20 == 0:
            print(f"  HTML images [{i}/{len(results)}]")
        cards.append(
            f"""
<article class="card" id="sample-{i}">
  <header>
    <span class="idx">#{i}</span>
    <span class="meta">{html.escape(r.get('folder', ''))} / {html.escape(r.get('file', ''))}</span>
    <span class="meta">p{r.get('page', '?')} · {html.escape(r.get('pdf_name', ''))}</span>
  </header>
  <div class="layout">
    <figure class="img-col">
      <img src="{html.escape(src)}" alt="{html.escape(r.get('id', ''))}" loading="lazy"/>
    </figure>
    <div class="text-cols">
      <section class="cap baseline">
        <h3>Baseline (pre-SFT, Qwen3-VL FP8)</h3>
        <p>{html.escape(r.get('baseline_caption') or '')}</p>
      </section>
      <section class="cap sft">
        <h3>SFT LoRA (Qwen2-VL-2B + adapter)</h3>
        <p>{html.escape(r.get('sft_caption') or '')}</p>
      </section>
      <section class="cap teacher">
        <h3>Teacher reference (qwen3.5-flash)</h3>
        <p>{html.escape(r.get('teacher_label') or '')}</p>
      </section>
    </div>
  </div>
</article>"""
        )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Baseline vs SFT — {len(results)} samples</title>
  <style>
    :root {{
      --bg: #0f1115; --card: #1a1d24; --border: #2a3140;
      --text: #e8eaed; --muted: #9aa0a6;
      --baseline: #5c7cfa; --sft: #51cf66; --teacher: #fcc419;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif;
      background: var(--bg); color: var(--text); line-height: 1.5;
    }}
    .top {{
      position: sticky; top: 0; z-index: 10;
      background: rgba(15,17,21,.92); backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--border); padding: 1rem 1.5rem;
    }}
    .top h1 {{ margin: 0 0 .35rem; font-size: 1.25rem; }}
    .top .sub {{ color: var(--muted); font-size: .875rem; }}
    .legend {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-top: .5rem; font-size: .8rem; }}
    .legend span {{ padding: .15rem .5rem; border-radius: 4px; }}
    .legend .b {{ background: rgba(92,124,250,.2); color: var(--baseline); }}
    .legend .s {{ background: rgba(81,207,102,.2); color: var(--sft); }}
    .legend .t {{ background: rgba(252,196,25,.2); color: var(--teacher); }}
    main {{ max-width: 1400px; margin: 0 auto; padding: 1rem 1.5rem 3rem; }}
    .card {{
      background: var(--card); border: 1px solid var(--border);
      border-radius: 10px; margin-bottom: 1.25rem; overflow: hidden;
    }}
    .card header {{
      padding: .6rem 1rem; border-bottom: 1px solid var(--border);
      display: flex; flex-wrap: wrap; gap: .5rem 1rem; align-items: baseline;
    }}
    .idx {{ font-weight: 700; color: var(--sft); }}
    .meta {{ color: var(--muted); font-size: .8rem; }}
    .layout {{ display: grid; grid-template-columns: minmax(200px, 340px) 1fr; gap: 0; }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }}
    }}
    .img-col {{
      margin: 0; padding: .75rem; background: #12151a;
      border-right: 1px solid var(--border);
      display: flex; align-items: flex-start; justify-content: center;
    }}
    .img-col img {{
      max-width: 100%; max-height: 320px; object-fit: contain;
      border-radius: 6px; background: #fff;
    }}
    .text-cols {{ display: flex; flex-direction: column; }}
    .cap {{
      padding: .75rem 1rem; border-bottom: 1px solid var(--border);
    }}
    .cap:last-child {{ border-bottom: none; }}
    .cap h3 {{
      margin: 0 0 .4rem; font-size: .75rem; text-transform: uppercase;
      letter-spacing: .04em; font-weight: 600;
    }}
    .cap.baseline h3 {{ color: var(--baseline); }}
    .cap.sft h3 {{ color: var(--sft); }}
    .cap.teacher h3 {{ color: var(--teacher); }}
    .cap p {{ margin: 0; font-size: .9rem; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <div class="top">
    <h1>Baseline vs SFT caption comparison</h1>
    <p class="sub">
      {len(results)} images · base <code>{html.escape(meta.get('base_model_id', ''))}</code>
      · adapter <code>{html.escape(meta.get('adapter_path', ''))}</code>
      · {html.escape(meta.get('generated_at', ''))}
    </p>
    <div class="legend">
      <span class="b">Baseline = pre-finetune 2B FP8 captions</span>
      <span class="s">SFT = LoRA fine-tuned Qwen2-VL-2B</span>
      <span class="t">Teacher = API reference (not shown to student at inference)</span>
    </div>
  </div>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT caption + HTML compare vs baseline")
    p.add_argument("--config", type=Path, default=QWEN_TEST / "config.yaml")
    p.add_argument(
        "--adapter",
        type=Path,
        default=QWEN_TEST / "output/sft/sft_lora_r16_ep2/adapter_final",
    )
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--records", type=Path, default=None)
    p.add_argument(
        "--lang",
        choices=("en", "ko"),
        default="en",
        help="en=vl_prompt + label/records; ko=vl_prompt_ko + label_ko/records",
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--dry-run", type=int, default=0, help="If >0, cap sample count for smoke test")
    p.add_argument(
        "--html-only",
        action="store_true",
        help="Regenerate HTML from existing JSON (no GPU)",
    )
    p.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="JSON for --html-only (default: output/sft/<run>/sft_captions_<limit>.json)",
    )
    p.add_argument(
        "--no-embed-images",
        action="store_true",
        help="Use relative file paths instead of base64 (may break on file:// + Unicode paths)",
    )
    p.add_argument("--thumb-width", type=int, default=480, help="Max width for embedded thumbnails")
    return p.parse_args()


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    args = parse_args()
    cfg = _load_config(args.config)
    embed_images = not args.no_embed_images

    adapter_dir = args.adapter.resolve()
    if not adapter_dir.is_dir():
        # fallback: user may have saved under Project/
        alt = Path("/home/sehun/Project/qwen_test/output/sft/sft_lora_r16_ep2/adapter_final")
        if alt.is_dir():
            adapter_dir = alt
        else:
            if not args.html_only:
                raise FileNotFoundError(f"Adapter not found: {args.adapter}")

    run_name = adapter_dir.parent.name
    out_dir = QWEN_TEST / "output" / "sft" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.html_only:
        json_path = args.from_json or (out_dir / f"sft_captions_{args.limit}.json")
        if not json_path.is_file():
            # pick largest sft_captions_*.json in out_dir
            candidates = sorted(out_dir.glob("sft_captions_*.json"), key=lambda p: p.stat().st_size)
            if not candidates:
                raise FileNotFoundError(f"No captions JSON in {out_dir}")
            json_path = candidates[-1]
        with json_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        results = payload["samples"]
        meta = payload.get("meta", {})
        n = len(results)
        html_path = out_dir / f"compare_baseline_vs_sft_{n}.html"
        print(f"📄 Rebuilding HTML from {json_path} ({n} samples, embed={embed_images})")
        _build_html(
            results,
            html_path,
            meta,
            embed_images=embed_images,
            thumb_max_width=args.thumb_width,
        )
        print(f"📁 {html_path}")
        print(f"   file://{html_path.resolve()}")
        return

    base_id = str(cfg.get("sft_model_id", "Qwen/Qwen2-VL-2B-Instruct"))
    if args.lang == "ko":
        prompt = str(cfg.get("vl_prompt_ko", "")).strip()
        records_path = args.records or (QWEN_TEST / "label_ko" / "records.jsonl")
    else:
        prompt = str(cfg.get("vl_prompt", "")).strip()
        records_path = args.records or (QWEN_TEST / "label" / "records.jsonl")
    limit = args.dry_run if args.dry_run > 0 else args.limit

    rows = _read_jsonl(records_path)
    samples = _select_samples(rows, limit, args.seed)
    print(f"📊 samples={len(samples)} (limit={limit}, seed={args.seed})")
    print(f"▶  adapter: {adapter_dir}")
    print(f"▶  base:    {base_id}")

    device = str(cfg.get("vl_cuda_device", args.gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    print("🔄 Loading model...")
    model, processor = _load_model(base_id, adapter_dir, device="0")

    print("🚀 Captioning...")
    t0 = time.perf_counter()
    results = _run_captioning(samples, model, processor, prompt, QWEN_TEST, args.max_new_tokens)
    total_sec = time.perf_counter() - t0

    n = len(results)
    meta = {
        "phase": "sft_post_finetune_compare",
        "base_model_id": base_id,
        "adapter_path": str(adapter_dir),
        "vl_prompt": prompt,
        "sample_count": n,
        "sample_seed": args.seed,
        "lang": args.lang,
        "source_records": str(records_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "caption_seconds": round(total_sec, 3),
        "baseline_source": "records.jsonl baseline_caption (Qwen3-VL-2B-FP8 pre-SFT)",
    }

    json_path = out_dir / f"sft_captions_{n}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"meta": meta, "samples": results}, f, ensure_ascii=False, indent=2)

    html_path = out_dir / f"compare_baseline_vs_sft_{n}.html"
    print(f"📄 Building HTML (embed_images={embed_images})...")
    _build_html(
        results,
        html_path,
        meta,
        embed_images=embed_images,
        thumb_max_width=args.thumb_width,
    )

    print(f"📁 {json_path}")
    print(f"📁 {html_path}")
    print(f"✅ Done in {total_sec:.1f}s — open HTML in browser:")
    print(f"   file://{html_path.resolve()}")


if __name__ == "__main__":
    main()
