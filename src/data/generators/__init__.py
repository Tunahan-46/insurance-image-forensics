"""Veri uretim fabrikasi (plan Hafta 2).

Moduller:
    prompts.py          Kombinatoryal prompt motoru (elle prompt YAZILMAZ)
    fully_synthetic.py  S katmani  - SD1.5 / SDXL / FLUX ile sifirdan uretim
    inpaint_add.py      M1 katmani - gercek fotografa olmayan hasar ekleme
    inpaint_remove.py   M2 katmani - var olan hasari silme
    classic_manip.py    M3 katmani - copy-move / splice / bg_replace (GPU'suz)

Ortak sozlesme: her uretici bir `GenResult` listesi doner ve hicbir uretici
manifest'e DOGRUDAN yazmaz. Manifest yazimi scripts/build_manifest_v2.py
icinde tek noktadadir; boylece dogrulama (sizinti kontrolu) atlanamaz.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenResult:
    """Bir uretim ciktisinin manifest'e girecek tum bilgisi.

    `source_image_id` KRITIKTIR: plan 4.5 Tuzak 1'e gore bir kaynak
    fotograftan turetilen HER SEY ayni split'te kalmalidir. Turetilmis
    goruntuler kaynagin id'sini tasir.
    """

    path: str
    label: str  # "real" | "fully_synthetic" | "manipulated"
    source_image_id: str
    manip_type: str = "none"
    generator: str = "none"
    mask_path: str = ""
    width: int = 0
    height: int = 0
    gen_params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Split kotasi
# ---------------------------------------------------------------------------

# Turetilmis goruntuler kaynak fotografin split'ini MIRAS ALIR (plan 4.5
# Tuzak 1). Kaynak havuzu tamamen rastgele gezilirse, uretilen orneklerin
# split dagilimi kaynagin dagilimini taklit eder:
#
#     CarDD gercek:  2854 train / 817 val / 381 test   ->  test payi %9.4
#
# Yani `--n 200` verildiginde test setine yalnizca ~19 ornek duser. W1'de
# aynisi yasandi (20 sentetigin %70/15/15 bolunmesi test'e 3 ornek
# birakmisti) ve tum sentetikler elle test'e alinmak zorunda kalindi.
#
# Test setinin ince olmasi, train'in ince olmasindan cok daha pahalidir:
# train Hafta 3'te buyutulebilir, ama test seti Cuma gunu DONDURULUYOR ve
# Hafta 5'e kadar acilmayacak. Ustelik tum metrikler (ROC-AUC, TPR@FPR,
# piksel-F1) onun uzerinden raporlanacak.
#
# Bu yuzden uretim, kaynagin dogal dagilimini degil, asagidaki kotayi
# hedefler. Kaynak split'leri DEGISMEZ -- degisen yalnizca hangi kaynak
# fotograflardan kac tane uretim yapildigidir.
SPLIT_QUOTA: dict[str, float] = {"train": 0.60, "val": 0.15, "test": 0.25}


def plan_by_split(
    pool, n: int, rng, quota: dict[str, float] | None = None
) -> tuple[list[int], dict[str, int]]:
    """Split kotasina gore isleme sirasi ve hedef sayilari uretir.

    Doner:
        order   -- pool icindeki indeksler; TEST ONCE gelir, cunku uretim
                   yarida kesilirse (Colab/Kaggle oturumu koparsa, kalite
                   kapisi cok reddederse) once en kritik split dolmus olur.
        targets -- split -> uretilecek ornek sayisi

    Cagiran, her basarili uretimde ilgili split sayacini artirmali ve
    kotasi dolan split'in satirlarini atlamalidir.
    """
    quota = quota or SPLIT_QUOTA
    splits = pool["split"].astype(str).to_numpy()

    targets: dict[str, int] = {}
    for name, frac in quota.items():
        available = int((splits == name).sum())
        want = int(round(n * frac))
        targets[name] = min(want, available)
        if want > available:
            print(
                f"  NOT: '{name}' split'inde {available} kaynak var, "
                f"{want} isteniyordu -- hedef {available}'e indirildi."
            )

    order: list[int] = []
    for name in ("test", "val", "train"):  # test once: bkz. docstring
        idxs = [i for i, s in enumerate(splits) if s == name]
        rng.shuffle(idxs)
        order.extend(idxs)

    return order, targets


__all__ = ["GenResult", "SPLIT_QUOTA", "plan_by_split"]
