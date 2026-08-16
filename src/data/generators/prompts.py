"""
Kombinatoryal prompt motoru (plan Hafta 2, "Prompt tasarimi").

MENTOR UYARISI (plan 4.3):
    "professional photography, 8k, cinematic" yazarsan model studyo
    kalitesinde gorsel uretir; dedektorun bunu ayirt etmesi trivial olur ve
    %99 accuracy'nin sebebi model degil PROMPT'UN olur. Bu, projeni
    gecersiz kilar.

Bu yuzden:
  - Pozitif prompt'lar ZORUNLU olarak bir "amator kalite" ve bir "kamera"
    ozelligi tasir (`build_prompt` bunu garanti eder, opsiyonel degil).
  - Negatif prompt her cagrida sabit ve zorunludur.
  - Her uretim kaydinin prompt/seed/model/steps/guidance degeri loglanir
    (reproducibility -- plan 4 altin kural).
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Kelime havuzlari
# ---------------------------------------------------------------------------

QUALITY = [
    "insurance claim photo",
    "smartphone photo",
    "amateur snapshot",
    "handheld photo",
    "dashcam-style still",
    "quick documentation photo",
]

COLOR = ["white", "silver", "black", "red", "dark blue", "grey", "beige", "dark green"]

VEHICLE = ["sedan", "hatchback", "SUV", "pickup truck", "minibus", "compact car", "station wagon"]

DAMAGE = [
    "deep scratch",
    "large dent",
    "shattered side window",
    "broken headlight",
    "crumpled bumper",
    "scraped paint",
    "cracked windshield",
    "flat tire",
    "torn off mirror",
]

PANEL = [
    "front bumper",
    "rear bumper",
    "driver side door",
    "passenger side door",
    "front fender",
    "hood",
    "tailgate",
    "left quarter panel",
    "right quarter panel",
]

SETTING = [
    "in a parking lot",
    "on a city street",
    "in a repair shop",
    "on a rainy road",
    "in an apartment garage",
    "on a residential side street",
    "in a supermarket car park",
]

LIGHT = [
    "overcast daylight",
    "direct sunlight",
    "fluorescent garage lighting",
    "golden hour",
    "flash at night",
    "dim evening light",
]

ANGLE = [
    "close-up 45 degree angle",
    "wide shot",
    "slightly tilted framing",
    "low angle close-up",
    "off-center framing",
]

CAMERA = [
    "shot on iPhone",
    "shot on Android phone, slight motion blur",
    "slightly out of focus",
    "handheld, minor camera shake",
    "phone camera, mild noise",
]

# Hasarsiz (negatif sinif) sentetikler icin -- "sentetik = hasarli" kestirme
# yolunu kapatir. Bu olmadan dedektor 'hasar var mi' sorusunu ogrenir,
# 'sentetik mi' sorusunu degil. Plan 4.5 Tuzak 4'un bir varyanti.
NO_DAMAGE_STATE = [
    "in good condition",
    "with no visible damage",
    "clean and undamaged",
]

NEGATIVE_PROMPT = (
    "professional, cinematic, 8k, ultra detailed, artstation, illustration, "
    "render, 3d, cartoon, anime, painting, oversaturated, studio lighting, "
    "perfect composition, watermark, text, logo, hdr, bokeh, depth of field"
)

# Inpainting (M1) icin bolgesel prompt'lar -- tum sahneyi degil, sadece
# maskelenmis paneli tarif ederler.
INPAINT_ADD_PROMPTS = [
    "deep scratch and dent on the car body panel",
    "crumpled dented metal on the car panel, scraped paint",
    "long deep scratch across the paint, exposed primer",
    "cracked and dented body panel with paint damage",
    "scraped and gouged car paint with visible dent",
]

INPAINT_REMOVE_PROMPTS = [
    "clean undamaged car body panel, smooth glossy paint",
    "intact car door panel, factory paint finish",
    "smooth undamaged metal panel, even color",
]


@dataclass(frozen=True)
class PromptSpec:
    """Uretilen bir prompt ve manifest'e yazilacak tum bilesenleri."""

    positive: str
    negative: str
    has_damage: bool
    damage: str
    panel: str
    vehicle: str
    color: str
    setting: str
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


def _pick(rng: np.random.Generator, pool: list[str]) -> str:
    return pool[int(rng.integers(0, len(pool)))]


def build_prompt(
    rng: np.random.Generator, *, damaged: bool = True, seed: int | None = None
) -> PromptSpec:
    """Tek bir prompt uretir.

    `quality` ve `camera` alanlari HER ZAMAN doldurulur -- bunlar opsiyonel
    degildir, cunku prompt naifligini engelleyen tek mekanizma bunlardir.
    """
    quality = _pick(rng, QUALITY)
    color = _pick(rng, COLOR)
    vehicle = _pick(rng, VEHICLE)
    setting = _pick(rng, SETTING)
    light = _pick(rng, LIGHT)
    angle = _pick(rng, ANGLE)
    camera = _pick(rng, CAMERA)

    if damaged:
        damage = _pick(rng, DAMAGE)
        panel = _pick(rng, PANEL)
        subject = f"{damage} on the {panel}"
    else:
        damage = ""
        panel = ""
        subject = _pick(rng, NO_DAMAGE_STATE)

    positive = (
        f"{quality} of a {color} {vehicle}, {subject}, "
        f"{setting}, {light}, {angle}, {camera}"
    )
    if seed is None:
        seed = int(rng.integers(0, 2**31 - 1))

    return PromptSpec(
        positive=positive,
        negative=NEGATIVE_PROMPT,
        has_damage=damaged,
        damage=damage,
        panel=panel,
        vehicle=vehicle,
        color=color,
        setting=setting,
        seed=seed,
    )


def build_prompt_batch(
    n: int, *, seed: int = 0, damaged_ratio: float = 0.75
) -> list[PromptSpec]:
    """n adet prompt uretir.

    damaged_ratio < 1.0 olmasi bilinclidir: sentetik katmanin bir kismi
    HASARSIZ araclardir. Aksi halde model "hasar goruyorsam sentetiktir"
    kestirmesini ogrenir ve gercek hasar fotograflarini yanlis siniflar --
    ki bu, sigorta baglaminda tam olarak istemedigimiz hata turudur.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        damaged = rng.random() < damaged_ratio
        out.append(build_prompt(rng, damaged=damaged))
    return out


def prompt_id(spec: PromptSpec) -> str:
    """Prompt + seed'den kisa deterministik kimlik -- dosya adi icin."""
    h = hashlib.md5(f"{spec.positive}|{spec.seed}".encode()).hexdigest()
    return h[:10]


def pick_inpaint_prompt(rng: np.random.Generator, mode: str) -> str:
    pool = INPAINT_ADD_PROMPTS if mode == "add" else INPAINT_REMOVE_PROMPTS
    return _pick(rng, pool)


def combination_space() -> int:
    """Teorik prompt cesitliligi -- dokumanda raporlanir."""
    return (
        len(QUALITY) * len(COLOR) * len(VEHICLE) * len(DAMAGE) * len(PANEL)
        * len(SETTING) * len(LIGHT) * len(ANGLE) * len(CAMERA)
    )


if __name__ == "__main__":
    print(f"Teorik kombinasyon sayisi: {combination_space():,}\n")

    batch = build_prompt_batch(6, seed=7)
    for i, s in enumerate(batch):
        tag = "HASARLI" if s.has_damage else "HASARSIZ"
        print(f"[{i}] ({tag}, seed={s.seed}, id={prompt_id(s)})")
        print(f"    {s.positive}\n")

    print(f"negative: {NEGATIVE_PROMPT}\n")

    # Kalite kapisi: yasakli kelimeler prompt'a sizmamali.
    banned = ["8k", "cinematic", "professional photography", "artstation", "ultra detailed"]
    big = build_prompt_batch(400, seed=1)
    for s in big:
        low = s.positive.lower()
        for b in banned:
            assert b not in low, f"YASAKLI KELIME prompt'a sizdi: '{b}' -> {s.positive}"
    assert all(any(c.split(",")[0].lower() in s.positive.lower() for c in CAMERA) for s in big[:20])

    n_damaged = sum(s.has_damage for s in big)
    print(f"400 prompt: {n_damaged} hasarli / {400 - n_damaged} hasarsiz")
    print(f"Benzersiz prompt: {len({s.positive for s in big})}/400")
    print("\nprompts.py sanity check OK")
