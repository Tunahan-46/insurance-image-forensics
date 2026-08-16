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


__all__ = ["GenResult"]
