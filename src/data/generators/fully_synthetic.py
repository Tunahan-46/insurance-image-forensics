"""
S katmani -- tam sentetik uretim (plan 4.2).

DIKKAT (plan 4.1): Tam sentetik uretim, gercek hayatta EN AZ OLASI ve
EN KOLAY tespit edilen saldiridir. Literaturun cogu bununla ilgilendigi
icin biz de bir taban olarak uretiyoruz, ama projenin agirlik merkezi
inpaint_add.py'dir. Bu dosyaya harcadigin sure sinirli olsun.

Cikti: data/raw/synthetic/{model}/{prompt_id}.png
       + ayni klasorde gen_log.jsonl (her satir bir uretim kaydi)

PNG olarak kaydediyoruz cunku laundering katmani JPEG'i ZATEN uyguluyor.
Burada da JPEG uygularsak cift sikistirma olur ve `clean` profili artik
"clean" olmaz. Ham cikti PNG, dagitima giren her sey laundering'den gecer.

Calistirma (Colab):
    python -m src.data.generators.fully_synthetic --model sdxl --n 400 --seed 1
    python -m src.data.generators.fully_synthetic --model sd15 --n 400 --seed 2
    python -m src.data.generators.fully_synthetic --model flux_schnell --n 400 --seed 3
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.data.generators import GenResult
from src.data.generators.pipelines import (
    MODEL_REGISTRY,
    RESOLUTION_POOL,
    load_txt2img,
    make_generator,
)
from src.data.generators.prompts import build_prompt_batch, prompt_id

DEFAULT_OUT = Path("data/raw/synthetic")


def generate(
    model: str,
    n: int,
    *,
    out_root: str | Path = DEFAULT_OUT,
    seed: int = 0,
    device: str = "cuda",
    damaged_ratio: float = 0.75,
    steps: int | None = None,
    guidance: float | None = None,
    resume: bool = True,
) -> list[GenResult]:
    """n adet tam sentetik goruntu uretir.

    `resume=True`: dosya zaten varsa atlanir. Colab oturumu koptugunda
    (ki gece boyu uretimde kopar) bastan baslamak zorunda kalmazsin.
    """
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Bilinmeyen model: {model}. Gecerli: {list(MODEL_REGISTRY)}")

    spec = MODEL_REGISTRY[model]
    steps = steps if steps is not None else spec.default_steps
    guidance = guidance if guidance is not None else spec.default_guidance

    out_dir = Path(out_root) / model
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "gen_log.jsonl"

    specs = build_prompt_batch(n, seed=seed, damaged_ratio=damaged_ratio)
    res_pool = RESOLUTION_POOL[model]
    rng = np.random.default_rng(seed)

    pipe = load_txt2img(model, device=device)
    results: list[GenResult] = []
    t0 = time.time()

    for i, ps in enumerate(specs):
        pid = prompt_id(ps)
        out_path = out_dir / f"{model}_{pid}.png"

        if resume and out_path.exists():
            continue

        w, h = res_pool[int(rng.integers(0, len(res_pool)))]
        kwargs = dict(
            prompt=ps.positive,
            width=w,
            height=h,
            num_inference_steps=steps,
            generator=make_generator(ps.seed, device),
        )
        # FLUX.1-schnell distilled'dir: negative prompt ve CFG kullanmaz.
        # Zorla gecirmek TypeError verir ya da sessizce yok sayilir.
        if model != "flux_schnell":
            kwargs["negative_prompt"] = ps.negative
            kwargs["guidance_scale"] = guidance
        else:
            kwargs["guidance_scale"] = 0.0

        image = pipe(**kwargs).images[0]
        image.save(out_path)

        record = {
            "path": str(out_path).replace("\\", "/"),
            "model": model,
            "repo": spec.repo,
            "steps": steps,
            "guidance": guidance,
            "width": w,
            "height": h,
            **ps.to_dict(),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        results.append(
            GenResult(
                path=record["path"],
                label="fully_synthetic",
                source_image_id=f"synth_{model}_{pid}",
                manip_type="none",
                generator=model,
                mask_path="",  # tam sentetikte 'manipule bolge' kavrami yok
                width=w,
                height=h,
                gen_params=record,
            )
        )

        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{n}  ({el/max(1,len(results)):.1f} sn/goruntu)")

    print(f"[{model}] {len(results)} yeni goruntu uretildi -> {out_dir}")
    return results


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Tam sentetik uretim")
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--damaged-ratio", type=float, default=0.75)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Model yuklemeden sadece prompt planini yazdir")
    a = ap.parse_args()

    if a.dry_run:
        specs = build_prompt_batch(min(a.n, 5), seed=a.seed, damaged_ratio=a.damaged_ratio)
        spec = MODEL_REGISTRY[a.model]
        print(f"model={a.model} repo={spec.repo} test_only={spec.test_only}")
        print(f"steps={a.steps or spec.default_steps} "
              f"guidance={a.guidance if a.guidance is not None else spec.default_guidance}")
        print(f"cozunurluk havuzu={RESOLUTION_POOL[a.model]}\n")
        for s in specs:
            print(f"  [{prompt_id(s)}] {s.positive}")
        return

    generate(
        a.model, a.n, out_root=a.out, seed=a.seed, device=a.device,
        damaged_ratio=a.damaged_ratio, steps=a.steps, guidance=a.guidance,
        resume=not a.no_resume,
    )


if __name__ == "__main__":
    _cli()
