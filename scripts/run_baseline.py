#!/usr/bin/env python3
"""
Extract image/figure/chart crops from PDFs (OCR_VL23 layout model) and caption with Qwen3-VL FP8.

Usage (ocr_opt env):
  python scripts/run_baseline.py                  # layout + caption all PDFs
  python scripts/run_baseline.py --layout-only
  python scripts/run_baseline.py --caption-only

Output:
  images/<pdf_folder>/p{page:03d}_b{box:02d}_{label}.jpg
  images/<pdf_folder>/manifest.json
  output/baseline_captions.json  (global index)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz
import numpy as np
import yaml
from PIL import Image

QWEN_TEST_DIR = Path(__file__).resolve().parent.parent
IMAGE_LABELS = frozenset({"image", "figure", "chart"})


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _slug_folder_name(pdf_path: Path) -> str:
    stem = pdf_path.stem.strip()
    s = re.sub(r"[^\w가-힣\-]+", "_", stem, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "pdf")[:120]


def _render_pdf(pdf_path: Path, dpi: float) -> Tuple[Any, List[np.ndarray]]:
    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    imgs: List[np.ndarray] = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        imgs.append(
            np.frombuffer(pix.samples, dtype=np.uint8)
            .reshape(pix.height, pix.width, 3)
            .copy()
        )
    return doc, imgs


def _crop_save(page_img: np.ndarray, coord: List[float], out_path: Path) -> bool:
    h, w = page_img.shape[:2]
    x1, y1, x2, y2 = (int(coord[0]), int(coord[1]), int(coord[2]), int(coord[3]))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(page_img[y1:y2, x1:x2]).save(out_path, format="JPEG", quality=90)
    return True


def _init_layout(ocr_vl_dir: Path, device: str):
    sys.path.insert(0, str(ocr_vl_dir))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import FormulaRecognitionPipeline

    yaml_path = str(ocr_vl_dir / "FormulaRecognitionPipeline.yaml")
    return FormulaRecognitionPipeline(paddlex_config=yaml_path, device=device)


def _run_layout(
    layout_model,
    doc,
    page_images: List[np.ndarray],
    dpi: float,
    batch_size: int,
    compute_layout_batch,
    layout_stage_config_from_environ,
) -> List[Dict[str, Any]]:
    cfg = layout_stage_config_from_environ()
    results: List[Dict[str, Any]] = []
    indices = list(range(len(page_images)))
    for s in range(0, len(indices), batch_size):
        batch_idx = indices[s : s + batch_size]
        batch_img = [page_images[i] for i in batch_idx]
        res_list, _ = compute_layout_batch(
            doc,
            batch_idx,
            dpi,
            batch_img,
            lambda imgs: list(layout_model.predict(imgs)),
            cfg,
        )
        results.extend(res_list)
    return results


def extract_pdf(
    pdf_path: Path,
    out_dir: Path,
    layout_model,
    *,
    dpi: float,
    layout_batch_size: int,
    compute_layout_batch,
    layout_stage_config_from_environ,
) -> Dict[str, Any]:
    folder = out_dir / _slug_folder_name(pdf_path)
    folder.mkdir(parents=True, exist_ok=True)

    doc, page_images = _render_pdf(pdf_path, dpi)
    t0 = time.perf_counter()
    layout_results = _run_layout(
        layout_model,
        doc,
        page_images,
        dpi,
        layout_batch_size,
        compute_layout_batch,
        layout_stage_config_from_environ,
    )
    dt_layout = time.perf_counter() - t0

    entries: List[Dict[str, Any]] = []
    for pi, (img, lay) in enumerate(zip(page_images, layout_results)):
        for bi, box in enumerate(lay.get("boxes", [])):
            label = str(box.get("label", "")).strip().lower()
            if label not in IMAGE_LABELS:
                continue
            coord = box.get("coordinate", [])
            if len(coord) != 4:
                continue
            fname = f"p{pi + 1:03d}_b{bi:02d}_{label}.jpg"
            fpath = folder / fname
            if not _crop_save(img, list(map(float, coord)), fpath):
                continue
            entries.append(
                {
                    "file": fname,
                    "page": pi + 1,
                    "page_index_0": pi,
                    "box_index": bi,
                    "label": label,
                    "bbox_px": [float(c) for c in coord],
                    "caption": None,
                }
            )

    doc.close()
    manifest = {
        "pdf": str(pdf_path.resolve()),
        "pdf_name": pdf_path.name,
        "folder": folder.name,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "dpi": dpi,
        "layout_seconds": round(dt_layout, 3),
        "image_count": len(entries),
        "images": entries,
    }
    with (folder / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"  ✅ {pdf_path.name}: {len(entries)} crops → {folder.name} ({dt_layout:.1f}s layout)")
    return manifest


def _resolve_model_path(model_id: str) -> str:
    from vlm_common import resolve_model_path

    resolved = resolve_model_path(model_id)
    if resolved != model_id:
        print(f"📦 Using local model: {resolved}")
    else:
        print(f"📦 Using HuggingFace model id: {model_id}")
    return resolved


def _auto_gpu_memory_utilization(cap: float = 0.85, floor: float = 0.52) -> float:
    """Fit vLLM reservation to currently free VRAM on cuda:0 (after CUDA_VISIBLE_DEVICES)."""
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            util = (free * 0.92) / total
            util = max(floor, min(cap, util))
            print(f"📊 GPU mem: free={free / 1e9:.2f}GB total={total / 1e9:.2f}GB → util={util:.2f}")
            return util
    except Exception:
        pass
    return float(cap)


def _init_vl(cfg: dict):
    backend = str(cfg.get("vl_backend", "vllm")).lower()
    model_path = _resolve_model_path(str(cfg.get("vl_model_id", "Qwen/Qwen3-VL-2B-Instruct-FP8")))
    if backend == "transformers":
        return _init_vl_transformers(model_path), None

    from vllm import LLM, SamplingParams

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_PLUGINS", "")
    cap = float(cfg.get("vl_gpu_memory_utilization", 0.85))
    util = _auto_gpu_memory_utilization(cap=cap, floor=0.35)
    try:
        llm = LLM(
            model=model_path,
            max_model_len=int(cfg.get("vl_max_model_len", 2048)),
            gpu_memory_utilization=util,
            tensor_parallel_size=1,
            dtype="auto",
            limit_mm_per_prompt={"image": 1},
            max_num_seqs=int(cfg.get("vl_max_num_seqs", 8)),
            enforce_eager=bool(cfg.get("vl_enforce_eager", True)),
            disable_log_stats=True,
        )
    except Exception as e:
        print(f"⚠️ vLLM init failed ({e}); falling back to transformers")
        return _init_vl_transformers(model_path), None
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    return llm, sp


def _init_vl_transformers(model_path: str):
    from vlm_common import init_vl_transformers, model_family

    fam = model_family(model_path)
    print(f"🔄 Loading model via transformers (family={fam})...")
    max_memory = {0: "14GiB", "cpu": "24GiB"} if fam == "qwen3_5" else {0: "7GiB", "cpu": "24GiB"}
    return init_vl_transformers(model_path, max_memory=max_memory)


def _caption_one_transformers(engine: dict, prompt: str, image_path: Path) -> str:
    import torch
    from PIL import Image

    model = engine["model"]
    processor = engine["processor"]
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    trimmed = [o[len(i) :] for i, o in zip(inputs["input_ids"], out_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def _image_to_b64(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def caption_folders(
    images_dir: Path,
    cfg: dict,
    *,
    llm=None,
    sp=None,
) -> Dict[str, Any]:
    prompt = str(cfg.get("vl_prompt", "")).strip()
    batch_size = max(1, int(cfg.get("vl_batch_size", 16)))
    model_id = str(cfg.get("vl_model_id", ""))

    folders = sorted(p for p in images_dir.iterdir() if p.is_dir())
    if not folders:
        print("No image folders found.")
        return {}

    own_llm = llm is None
    if own_llm:
        print("🔄 Loading Qwen3-VL FP8 for captioning...")
        llm, sp = _init_vl(cfg)

    assert llm is not None
    use_tf = isinstance(llm, dict) and llm.get("backend") == "transformers"
    if not use_tf and sp is None:
        raise RuntimeError("vLLM SamplingParams missing")
    total_captioned = 0
    t_all = time.perf_counter()

    for folder in folders:
        manifest_path = folder / "manifest.json"
        if not manifest_path.exists():
            print(f"  ⚠️ skip {folder.name}: no manifest.json")
            continue
        with manifest_path.open(encoding="utf-8") as f:
            manifest = json.load(f)

        jobs: List[Tuple[Dict[str, Any], Path]] = []
        for ent in manifest.get("images", []):
            rel = ent.get("file")
            if not rel:
                continue
            img_path = folder / rel
            if img_path.is_file():
                jobs.append((ent, img_path))

        if not jobs:
            continue

        convs, keys = [], []
        for ent, img_path in jobs:
            b64 = _image_to_b64(img_path)
            convs.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                        ],
                    }
                ]
            )
            keys.append(ent)

        captions: Dict[str, str] = {}
        t0 = time.perf_counter()
        if use_tf:
            for ent, img_path in jobs:
                cap = _caption_one_transformers(llm, prompt, img_path)
                ent["caption"] = cap
                captions[ent["file"]] = cap
        else:
            for s in range(0, len(convs), batch_size):
                chunk = convs[s : s + batch_size]
                chunk_keys = keys[s : s + batch_size]
                outs = llm.chat(chunk, sampling_params=sp, use_tqdm=False)
                for ent, out in zip(chunk_keys, outs):
                    cap = out.outputs[0].text.strip()
                    ent["caption"] = cap
                    captions[ent["file"]] = cap

        manifest["caption_model_id"] = model_id
        manifest["vl_prompt"] = prompt
        manifest["captioned_at"] = datetime.now(timezone.utc).isoformat()
        manifest["caption_seconds"] = round(time.perf_counter() - t0, 3)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        total_captioned += len(jobs)
        print(f"  ✅ captioned {folder.name}: {len(jobs)} images ({manifest['caption_seconds']}s)")

    summary = {
        "phase": "baseline_pre_finetune",
        "model_id": model_id,
        "vl_prompt": prompt,
        "captioned_at": datetime.now(timezone.utc).isoformat(),
        "total_images": total_captioned,
        "total_seconds": round(time.perf_counter() - t_all, 3),
        "folders": [],
    }
    for folder in folders:
        mp = folder / "manifest.json"
        if not mp.exists():
            continue
        with mp.open(encoding="utf-8") as f:
            m = json.load(f)
        summary["folders"].append(
            {
                "folder": folder.name,
                "pdf_name": m.get("pdf_name"),
                "image_count": m.get("image_count"),
                "images": m.get("images", []),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract layout crops and baseline VL captions")
    parser.add_argument("--config", type=Path, default=QWEN_TEST_DIR / "config.yaml")
    parser.add_argument("--layout-only", action="store_true")
    parser.add_argument("--caption-only", action="store_true")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA device for vLLM")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    ocr_vl_dir = Path(cfg.get("ocr_vl_dir", "")).resolve()
    pdf_dir = Path(cfg.get("pdf_dir", QWEN_TEST_DIR / "pdf")).resolve()
    images_dir = Path(cfg.get("images_dir", QWEN_TEST_DIR / "images")).resolve()
    output_dir = Path(cfg.get("output_dir", QWEN_TEST_DIR / "output")).resolve()
    images_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    do_layout = not args.caption_only
    do_caption = not args.layout_only

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if do_layout and not pdfs:
        print(f"No PDFs in {pdf_dir}")
        sys.exit(1)

    if do_layout:
        sys.path.insert(0, str(ocr_vl_dir))
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        from processing.layout_extract import compute_layout_batch, layout_stage_config_from_environ

        dpi = float(cfg.get("dpi", 92))
        layout_bs = int(cfg.get("layout_batch_size", 16))
        layout_device = str(cfg.get("layout_device", "gpu:0"))

        print(f"🔄 Loading layout model ({layout_device})...")
        layout_model = _init_layout(ocr_vl_dir, layout_device)
        print(f"📄 Processing {len(pdfs)} PDFs → {images_dir}")

        all_manifests: List[Dict[str, Any]] = []
        for pdf in pdfs:
            try:
                m = extract_pdf(
                    pdf,
                    images_dir,
                    layout_model,
                    dpi=dpi,
                    layout_batch_size=layout_bs,
                    compute_layout_batch=compute_layout_batch,
                    layout_stage_config_from_environ=layout_stage_config_from_environ,
                )
                all_manifests.append(m)
            except Exception as e:
                print(f"  ❌ {pdf.name}: {e}")

        extract_index = {
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "pdf_count": len(all_manifests),
            "total_images": sum(m.get("image_count", 0) for m in all_manifests),
            "manifests": [
                {"folder": m["folder"], "pdf_name": m["pdf_name"], "image_count": m["image_count"]}
                for m in all_manifests
            ],
        }
        with (output_dir / "extract_index.json").open("w", encoding="utf-8") as f:
            json.dump(extract_index, f, ensure_ascii=False, indent=2)
        print(f"📁 extract_index.json written ({extract_index['total_images']} images)")

    if do_caption:
        gpu = str(cfg.get("vl_cuda_device", args.gpu))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        print(
            f"🔄 Captioning (CUDA_VISIBLE_DEVICES={gpu}, "
            f"mem_util={cfg.get('vl_gpu_memory_utilization')})..."
        )
        summary = caption_folders(images_dir, cfg)
        out_path = output_dir / "baseline_captions.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"📁 {out_path} ({summary.get('total_images', 0)} captions)")


if __name__ == "__main__":
    main()
