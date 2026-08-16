"""
M1 katmani -- GERCEK fotografa OLMAYAN hasar ekleme (plan tablo 4.1, #1).

Bu, projenin EN KRITIK senaryosudur:
    gercek hayatta olasilik: cok yuksek
    tespit zorlugu        : zor
    literaturde ilgi      : dusuk

Iki alt tip:
    M1a  damage_add      -> hasarsiz panele yeni hasar (maske = eklenen bolge)
    M1b  damage_enlarge  -> mevcut hasari buyutme (maske = HALKA, yeni \\ eski)

M1b, tablo 4.1'e gore tespiti EN ZOR saldiridir cunku goruntunun buyuk
kismi gercektir ve manipule bolge gercek bir hasarin hemen bitisigindedir.

VERI SIZINTISI KURALI
---------------------
Cikti `source_image_id` olarak KAYNAK CarDD goruntusunun id'sini tasir
(`cardd_XXXX`). Boylece manifest'teki gruplu split, gercek foto ile ondan
turetilmis manipulasyonu ayni tarafta tutar (plan 4.5 Tuzak 1).

Cikti:
    data/raw/manipulated/inpaint_add/{sid}_{tag}.png
    data/raw/manipulated/inpaint_add/masks/{sid}_{tag}.png
    gen_log.jsonl

Calistirma (Colab):
    python -m src.data.generators.inpaint_add --manifest data/processed/manifest_v1.parquet \\
        --n 800 --model sdxl --seed 11
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
from src.data.generators.pipelines import MODEL_REGISTRY, load_inpaint, make_generator
from src.data.generators.prompts import NEGATIVE_PROMPT, pick_inpaint_prompt
from src.data.masks import (
    changed_fraction_in_mask,
    leak_fraction_outside_mask,
    load_mask,
    mask_area_frac,
    mask_for_damage_add,
    mask_for_damage_enlarge,
    save_mask,
    soft_mask,
)

DEFAULT_OUT = Path("data/raw/manipulated/inpaint_add")

# Kabul kapisi: inpaint modeli maskeyi neredeyse aynen yeniden uretmis
# olabilir (ozellikle dusuk strength ile). O zaman "manipule" etiketi
# yalan olur. Bkz. src.data.masks.changed_fraction_in_mask.
MIN_CHANGED_IN_MASK = 0.30

# Inpaint hatlari sabit boyut ister. Kaynak goruntu bu boyuta getirilir,
# uretim sonrasi ORIJINAL boyuta geri dondurulur -- aksi halde tum
# manipule goruntuler ayni cozunurlukte olur ve plan 4.5 Tuzak 2'yi
# (model cozunurlugu ogrenir) kendi elimizle kurmus oluruz.
WORK_SIZE = {"sd15": 512, "sdxl": 1024}


def _to_work(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)


def _blend_back(
    original_bgr: np.ndarray, generated_bgr: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Uretilen bolgeyi orijinal goruntuye yumusak maske ile geri harmanlar.

    NEDEN sadece pipe ciktisini kullanmiyoruz: SD inpaint TUM goruntuyu
    yeniden encode/decode eder (VAE round-trip). Bu, maske disindaki her
    pikseli de degistirir ve zemin gercegi maskesi YALAN olur -- dedektor
    "manipulasyon her yerde" der, localization metrigi anlamsizlasir.
    Sadece maske icini geri yazmak zemin gercegini dogru tutar.
    """
    a = (soft_mask(mask, 21).astype(np.float32) / 255.0)[..., None]
    return (generated_bgr.astype(np.float32) * a
            + original_bgr.astype(np.float32) * (1 - a)).astype(np.uint8)


def _damage_mask_path_from_manifest(gen_params: str) -> str:
    try:
        return json.loads(gen_params).get("damage_mask_path", "")
    except (json.JSONDecodeError, TypeError):
        return ""


def generate(
    manifest_path: str | Path,
    n: int,
    *,
    model: str = "sdxl",
    out_root: str | Path = DEFAULT_OUT,
    seed: int = 0,
    device: str = "cuda",
    enlarge_ratio: float = 0.35,
    strength: float = 0.95,
    steps: int | None = None,
    guidance: float | None = None,
    resume: bool = True,
) -> list[GenResult]:
    """Manifest'teki CarDD gercek goruntulerinden n adet M1 ornegi uretir.

    enlarge_ratio: kacinin M1b (hasar buyutme) olacagi. Varsayilan 0.35 --
    M1b daha zor ve daha degerli, ama her goruntude uygun hasar maskesi
    olmayabilir, o yuzden cogunluk M1a.
    """
    from src.data.manifest import load_manifest

    spec = MODEL_REGISTRY[model]
    if spec.test_only:
        raise ValueError(
            f"{model} test-only bir ureticidir ve manipulasyon katmaninda "
            f"kullanilamaz (E6 'gorulmemis uretici' deneyini kirar)."
        )
    steps = steps if steps is not None else spec.default_steps
    guidance = guidance if guidance is not None else spec.default_guidance
    work = WORK_SIZE[model]

    out_dir = Path(out_root)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "gen_log.jsonl"

    df = load_manifest(manifest_path)
    pool = df[(df["label"] == "real") & (df["launder_profile"] == "clean")]
    if len(pool) == 0:
        raise RuntimeError("Manifest'te 'real' + 'clean' goruntu yok.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))

    pipe = load_inpaint(model, device=device)
    results: list[GenResult] = []
    t0 = time.time()
    attempted = skipped_no_mask = rejected = 0

    for idx in order:
        if len(results) >= n:
            break
        row = pool.iloc[int(idx)]
        attempted += 1

        src_path = Path(row["path"])
        if not src_path.exists():
            continue

        img_bgr = imread(src_path)
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]

        dmg_path = _damage_mask_path_from_manifest(row["gen_params"])
        dmg = load_mask(dmg_path, size=(W, H)) if dmg_path and Path(dmg_path).exists() else None

        do_enlarge = dmg is not None and rng.random() < enlarge_ratio
        if do_enlarge:
            manip_mask = mask_for_damage_enlarge(dmg, rng=rng)
            manip_type = "inpaint_enlarge"
        else:
            manip_mask = mask_for_damage_add(img_bgr, dmg, rng=rng)
            manip_type = "inpaint_add"

        if manip_mask is None:
            skipped_no_mask += 1
            continue

        sid = str(row["source_image_id"])
        tag = f"{manip_type}_{model}"
        out_path = out_dir / f"{sid}_{tag}.png"
        mask_path = out_dir / "masks" / f"{sid}_{tag}.png"
        if resume and out_path.exists() and mask_path.exists():
            continue

        prompt = pick_inpaint_prompt(rng, "add")
        gen_seed = int(rng.integers(0, 2**31 - 1))

        img_w = _to_work(img_bgr, work)
        msk_w = _to_work(manip_mask, work)

        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=Image.fromarray(cv2.cvtColor(img_w, cv2.COLOR_BGR2RGB)),
            mask_image=Image.fromarray(msk_w),
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=strength,
            generator=make_generator(gen_seed, device),
        ).images[0]

        gen_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        gen_bgr = cv2.resize(gen_bgr, (W, H), interpolation=cv2.INTER_LANCZOS4)
        final = _blend_back(img_bgr, gen_bgr, manip_mask)

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
            "manip_type": manip_type,
            "model": model,
            "repo": spec.inpaint_repo,
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": gen_seed,
            "steps": steps,
            "guidance": guidance,
            "strength": strength,
            "work_size": work,
            "mask_area_frac": round(mask_area_frac(manip_mask), 5),
            "changed_frac_in_mask": round(changed, 4),
            "leak_frac_outside_mask": round(leak, 5),
            "had_damage_mask": dmg is not None,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        results.append(
            GenResult(
                path=record["path"],
                label="manipulated",
                source_image_id=sid,  # <- KAYNAGIN id'si; sizinti korumasi
                manip_type=manip_type,
                generator=model,
                mask_path=record["mask_path"],
                width=W,
                height=H,
                gen_params=record,
            )
        )

        if len(results) % 20 == 0:
            print(f"  {len(results)}/{n}  ({(time.time()-t0)/len(results):.1f} sn/goruntu)")

    print(
        f"[inpaint_add/{model}] {len(results)} uretildi | {attempted} denendi | "
        f"{skipped_no_mask} uygun maske yok | {rejected} kalite kapisinda reddedildi"
    )
    if rejected > 0.3 * max(1, len(results)):
        print(
            "  UYARI: Reddetme orani yuksek. strength degerini artir "
            "(varsayilan 0.95) veya guidance'i yukselt -- model maskeyi "
            "neredeyse aynen yeniden uretiyor."
        )
    return results


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="M1: hasar ekleme / buyutme")
    ap.add_argument("--manifest", default="data/processed/manifest_v1.parquet")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--model", default="sdxl", choices=["sd15", "sdxl"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--enlarge-ratio", type=float, default=0.35)
    ap.add_argument("--strength", type=float, default=0.95)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    generate(
        a.manifest, a.n, model=a.model, out_root=a.out, seed=a.seed,
        device=a.device, enlarge_ratio=a.enlarge_ratio, strength=a.strength,
        steps=a.steps, guidance=a.guidance, resume=not a.no_resume,
    )


if __name__ == "__main__":
    _cli()
