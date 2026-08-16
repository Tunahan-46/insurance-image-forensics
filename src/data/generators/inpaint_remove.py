"""
M2 katmani -- var olan hasari / nesneyi SILME (plan tablo 4.1, #4).

Sigorta baglaminda iki yonu var:
  - Hasar silme  : "bu hasar eskiden yoktu" iddiasi icin temiz 'once' foto
  - Damga silme  : tarih damgasi, plaka, park cezasi etiketi silme

Kaynak: CarDD gercek fotograflari + hazir hasar maskeleri (W1 Bulgu 2).
Zemin gercegi maskesi = silinen bolge.

IKI YONTEM
----------
  sd_inpaint : SD/SDXL inpaint, prompt="clean undamaged car body panel"
  telea/ns   : OpenCV klasik inpaint -- GPU'suz, hizli, dusuk kalite

Klasik yontemi BILINCLI olarak tutuyoruz. LaMa kurmak Colab'da bagimlilik
derdi; OpenCV inpaint ise gercek dunyada da kullanilan (Snapseed "healing",
Photoshop "spot healing" akrabasi) ve FARKLI bir forensic iz birakan bir
yontem. Dedektorun sadece difuzyon izini degil, klasik doldurma izini de
gormesi genellemeyi artirir.

Cikti:
    data/raw/manipulated/inpaint_remove/{sid}_{tag}.png
    data/raw/manipulated/inpaint_remove/masks/{sid}_{tag}.png

Calistirma:
    # GPU'lu (Colab)
    python -m src.data.generators.inpaint_remove --n 300 --method sd_inpaint --model sdxl
    # GPU'suz (yerel, hemen calisir)
    python -m src.data.generators.inpaint_remove --n 100 --method telea --device cpu
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.data.generators import GenResult
from src.data.imageio import imread, imwrite
from src.data.generators.inpaint_add import WORK_SIZE, _blend_back, _to_work
from src.data.generators.pipelines import MODEL_REGISTRY, load_inpaint, make_generator
from src.data.generators.prompts import NEGATIVE_PROMPT, pick_inpaint_prompt
from src.data.masks import (
    changed_fraction_in_mask,
    leak_fraction_outside_mask,
    load_mask,
    mask_area_frac,
    mask_for_removal,
    save_mask,
)

DEFAULT_OUT = Path("data/raw/manipulated/inpaint_remove")
METHODS = ("sd_inpaint", "telea", "ns")

# Kabul kapisi -- bkz. src.data.masks.changed_fraction_in_mask.
# Silme senaryosunda esik biraz daha dusuk: hasar zaten kucuk kontrastli
# olabilir ve dogru bir silme az piksel degistirir.
MIN_CHANGED_IN_MASK = 0.20

CV2_FLAGS = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}


def _classic_inpaint(img_bgr: np.ndarray, mask: np.ndarray, method: str) -> np.ndarray:
    """OpenCV inpaint. radius=7: kucuk degerler yamayi belli eder, buyuk
    degerler bulanik leke birakir; 5-9 araligi tipik hasar boyutlari icin
    dengeli."""
    return cv2.inpaint(img_bgr, (mask > 127).astype(np.uint8), 7, CV2_FLAGS[method])


def generate(
    manifest_path: str | Path,
    n: int,
    *,
    method: str = "sd_inpaint",
    model: str = "sdxl",
    out_root: str | Path = DEFAULT_OUT,
    seed: int = 0,
    device: str = "cuda",
    steps: int | None = None,
    guidance: float | None = None,
    strength: float = 0.9,
    resume: bool = True,
) -> list[GenResult]:
    if method not in METHODS:
        raise ValueError(f"Bilinmeyen method: {method}. Gecerli: {METHODS}")

    from src.data.generators.inpaint_add import _damage_mask_path_from_manifest
    from src.data.manifest import load_manifest

    out_dir = Path(out_root)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "gen_log.jsonl"

    df = load_manifest(manifest_path)
    pool = df[(df["label"] == "real") & (df["launder_profile"] == "clean")]
    # M2 icin hasar maskesi ZORUNLU -- silinecek bir sey olmali.
    pool = pool[pool["gen_params"].astype(str).str.contains("damage_mask_path")]
    if len(pool) == 0:
        raise RuntimeError(
            "Hasar maskesi referansi olan 'real' satir yok. "
            "Once scripts/build_manifest_v1.py calistir."
        )

    pipe = None
    if method == "sd_inpaint":
        spec = MODEL_REGISTRY[model]
        if spec.test_only:
            raise ValueError(f"{model} test-only; manipulasyon katmaninda kullanilamaz.")
        steps = steps if steps is not None else spec.default_steps
        guidance = guidance if guidance is not None else spec.default_guidance
        pipe = load_inpaint(model, device=device)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    results: list[GenResult] = []
    t0 = time.time()
    skipped = 0
    rejected = 0

    for idx in order:
        if len(results) >= n:
            break
        row = pool.iloc[int(idx)]
        src_path = Path(row["path"])
        dmg_path = _damage_mask_path_from_manifest(row["gen_params"])
        if not src_path.exists() or not dmg_path or not Path(dmg_path).exists():
            skipped += 1
            continue

        img_bgr = imread(src_path)
        if img_bgr is None:
            skipped += 1
            continue
        H, W = img_bgr.shape[:2]

        dmg = load_mask(dmg_path, size=(W, H))
        manip_mask = mask_for_removal(dmg, rng=rng)
        if manip_mask is None or mask_area_frac(manip_mask) > 0.35:
            # Cok buyuk maske: inpaint "yeni bir araba" uydurur, senaryo
            # gercekci olmaktan cikar.
            skipped += 1
            continue

        sid = str(row["source_image_id"])
        tag = f"remove_{method}" + (f"_{model}" if method == "sd_inpaint" else "")
        out_path = out_dir / f"{sid}_{tag}.png"
        mask_path = out_dir / "masks" / f"{sid}_{tag}.png"
        if resume and out_path.exists() and mask_path.exists():
            continue

        gen_seed = int(rng.integers(0, 2**31 - 1))
        prompt = ""

        if method == "sd_inpaint":
            work = WORK_SIZE[model]
            prompt = pick_inpaint_prompt(rng, "remove")
            img_w = _to_work(img_bgr, work)
            msk_w = _to_work(manip_mask, work)
            out = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                image=Image.fromarray(cv2.cvtColor(img_w, cv2.COLOR_BGR2RGB)),
                mask_image=Image.fromarray(msk_w),
                num_inference_steps=steps,
                guidance_scale=guidance,
                strength=strength,
                generator=make_generator(gen_seed, device),
            ).images[0]
            gen_bgr = cv2.resize(
                cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR), (W, H),
                interpolation=cv2.INTER_LANCZOS4,
            )
            final = _blend_back(img_bgr, gen_bgr, manip_mask)
        else:
            final = _classic_inpaint(img_bgr, manip_mask, method)

        changed = changed_fraction_in_mask(img_bgr, final, manip_mask)
        if changed < MIN_CHANGED_IN_MASK:
            rejected += 1
            continue
        leak = leak_fraction_outside_mask(img_bgr, final, manip_mask)

        imwrite(out_path, final)
        save_mask(manip_mask, mask_path)

        record = {
            "path": str(out_path).replace("\\", "/"),
            "mask_path": str(mask_path).replace("\\", "/"),
            "source_path": str(src_path).replace("\\", "/"),
            "manip_type": "inpaint_remove",
            "method": method,
            "model": model if method == "sd_inpaint" else "opencv",
            "prompt": prompt,
            "seed": gen_seed,
            "steps": steps,
            "guidance": guidance,
            "mask_area_frac": round(mask_area_frac(manip_mask), 5),
            "changed_frac_in_mask": round(changed, 4),
            "leak_frac_outside_mask": round(leak, 5),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        results.append(
            GenResult(
                path=record["path"],
                label="manipulated",
                source_image_id=sid,
                manip_type="inpaint_remove",
                generator=record["model"],
                mask_path=record["mask_path"],
                width=W,
                height=H,
                gen_params=record,
            )
        )

        if len(results) % 20 == 0:
            print(f"  {len(results)}/{n}  ({(time.time()-t0)/len(results):.2f} sn/goruntu)")

    print(
        f"[inpaint_remove/{method}] {len(results)} uretildi | {skipped} atlandi | "
        f"{rejected} kalite kapisinda reddedildi"
    )
    return results


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="M2: hasar/nesne silme")
    ap.add_argument("--manifest", default="data/processed/manifest_v1.parquet")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--method", default="sd_inpaint", choices=METHODS)
    ap.add_argument("--model", default="sdxl", choices=["sd15", "sdxl"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--strength", type=float, default=0.9)
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    generate(
        a.manifest, a.n, method=a.method, model=a.model, out_root=a.out,
        seed=a.seed, device=a.device, steps=a.steps, guidance=a.guidance,
        strength=a.strength, resume=not a.no_resume,
    )


if __name__ == "__main__":
    _cli()
