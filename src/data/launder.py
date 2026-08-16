"""
Laundering katmani (plan bolum 4.4 ve Hafta 2).

Bu modul projenin en ozgun deneysel katkisidir. Literaturdeki dedektorlerin
cogu `clean` goruntuler uzerinde raporlanir; sigorta gercegi ise `whatsapp`tir.
Her test goruntusu 5 profilde de degerlendirilir ve sonuc tablosu
senaryo x profil matrisi olur.

TASARIM KARARLARI
-----------------
1. Profiller SAF FONKSIYON zinciridir. Girdi PIL.Image, cikti PIL.Image.
   Yan etkisi yoktur, boylece test edilebilir ve GPU gerektirmez.

2. Diske yazma tek noktadan (`launder_file`) yapilir. Cunku JPEG kalitesi
   ancak kaydederken uygulanabilir; zincirin son adimi her zaman bir
   encode islemidir ve bunun tek yerde olmasi profillerin karsilastirilabilir
   kalmasini saglar.

3. `clean` profili de YENIDEN KAYDEDILIR (q=95). Bu bilincli bir karardir:
   plan 4.5 Tuzak 3 -- "real'ler JPEG, fake'ler PNG ise model formati ogrenir".
   Tum goruntuler ayni encoder'dan gecerse bu kestirme yol kapanir.
   Metadata dedektoru (L1) HER ZAMAN orijinal dosyadan okumalidir; bu yuzden
   laundered dosyalar L1 icin degil, L2/L3 gorsel katmanlari icindir.

4. EXIF laundering sirasinda BILEREK dusurulur. Gercek hayatta WhatsApp da
   dusurur. Orijinal EXIF, manifest'in `clean` satirindaki orijinal path'te
   ve `exif_json` yan dosyasinda korunur (bkz. scripts/ingest_own_photos.py).

5. Maskeler laundering'den ETKILENMEZ ama YENIDEN BOYUTLANDIRILIR.
   Goruntu 1600px'e kuculduyse maske de kuculmeli, aksi halde piksel-F1
   hesabi sacmalar. `launder_mask` bunu nearest-neighbour ile yapar.

Kullanim:
    python -m src.data.launder --demo
    python scripts/apply_laundering.py --manifest data/processed/manifest_v1.parquet
"""
from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter

LaunderProfile = Literal["clean", "whatsapp", "screenshot", "double_jpeg", "aggressive"]

PROFILE_NAMES: tuple[str, ...] = (
    "clean",
    "whatsapp",
    "screenshot",
    "double_jpeg",
    "aggressive",
)

# Egitimde augmentation olarak kullanilacak profiller (plan Hafta 2, split kurali 3).
# Test'te BES profilin HEPSI ayri ayri raporlanir.
TRAIN_AUGMENT_PROFILES: tuple[str, ...] = ("clean", "whatsapp", "double_jpeg")


# ---------------------------------------------------------------------------
# Atomik islemler
# ---------------------------------------------------------------------------


def resize_long_edge(img: Image.Image, target: int) -> Image.Image:
    """Uzun kenari `target` olacak sekilde orana sadik kalarak kucultur.
    Goruntu zaten daha kucukse DOKUNULMAZ -- yapay upscale, olmayan bir
    interpolasyon izi ekler ve dedektore sahte sinyal verir."""
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= target:
        return img
    scale = target / long_edge
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(new_size, Image.LANCZOS)


def jpeg_roundtrip(img: Image.Image, quality: int, subsampling: int = 2) -> Image.Image:
    """Bellekte JPEG encode/decode. subsampling=2 => 4:2:0, telefon/mesajlasma
    uygulamalarinin varsayilani. Kroma alt-ornekleme JPEG izinin buyuk kismini
    olusturur; bunu atlarsak laundering gercekci olmaz."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    out = Image.open(buf)
    out.load()  # buffer kapanmadan pikselleri belleğe al
    return out


def png_roundtrip(img: Image.Image) -> Image.Image:
    """PNG uzerinden gecir. Kayipsizdir; tek etkisi onceki JPEG blok yapisini
    'dondurup' sonraki JPEG'in farkli bir grid ile ustune binmesidir --
    ekran goruntusu almanin tam olarak yaptigi sey budur."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out


def gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def screenshot_crop(img: Image.Image, frac: float = 0.02) -> Image.Image:
    """Ekran goruntusu alirken kenarlardan bir miktar kirpilir. Bu, JPEG
    8x8 blok gridini KAYDIRIR -- double-JPEG tespitini zorlastiran ana
    faktordur ve gercekci senaryonun parcasidir."""
    w, h = img.size
    dx, dy = int(w * frac), int(h * frac)
    if w - 2 * dx < 32 or h - 2 * dy < 32:
        return img
    return img.crop((dx, dy, w - dx, h - dy))


# ---------------------------------------------------------------------------
# Profiller
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    """Bir laundering profili: ara adimlar + son kaydetme kalitesi."""

    name: str
    steps: tuple[Callable[[Image.Image], Image.Image], ...]
    save_quality: int
    description: str
    # Goruntu kirpiliyorsa maske de ayni sekilde kirpilmali.
    crops: bool = False

    def apply(self, img: Image.Image) -> Image.Image:
        for step in self.steps:
            img = step(img)
        return img


LAUNDER_PROFILES: dict[str, Profile] = {
    "clean": Profile(
        name="clean",
        steps=(),
        save_quality=95,
        description="Laboratuvar kosulu. Yalnizca q95 JPEG olarak yeniden kaydedilir "
        "(plan 4.5 Tuzak 3: format kestirme yolunu kapatmak icin).",
    ),
    "whatsapp": Profile(
        name="whatsapp",
        steps=(lambda im: resize_long_edge(im, 1600),),
        save_quality=75,
        description="WhatsApp gonderimi: uzun kenar 1600px + q75. "
        "Sigorta dosyalarinda EN SIK karsilasilan durum.",
    ),
    "screenshot": Profile(
        name="screenshot",
        steps=(
            lambda im: resize_long_edge(im, 1280),
            screenshot_crop,
            png_roundtrip,
        ),
        save_quality=90,
        description="Ekran goruntusu alip gonderme: resize + kirpma (JPEG grid kayar) "
        "+ PNG ara adimi + q90.",
        crops=True,
    ),
    "double_jpeg": Profile(
        name="double_jpeg",
        steps=(lambda im: jpeg_roundtrip(im, 95),),
        save_quality=70,
        description="Kaydet-duzenle-kaydet: q95 -> q70. Ayni grid uzerinde ikili "
        "sikistirma; klasik DQ-effect burada gorulur.",
    ),
    "aggressive": Profile(
        name="aggressive",
        steps=(
            lambda im: resize_long_edge(im, 1024),
            lambda im: gaussian_blur(im, 0.5),
        ),
        save_quality=60,
        description="En kotu senaryo: 1024px + q60 + hafif blur. Yuksek frekansli "
        "forensic izlerin cogu burada yok olur -- dedektorun alt sinirini olcer.",
    ),
}


# ---------------------------------------------------------------------------
# Dosya seviyesi API
# ---------------------------------------------------------------------------


def launder_image(img: Image.Image, profile: str) -> tuple[Image.Image, Profile]:
    """PIL goruntuye profili uygular. Kaydetme YAPMAZ -- son JPEG encode
    `launder_file` icindedir (bkz. modul basligi, tasarim karari 2)."""
    if profile not in LAUNDER_PROFILES:
        raise ValueError(f"Bilinmeyen profil: {profile}. Gecerli: {PROFILE_NAMES}")
    p = LAUNDER_PROFILES[profile]
    return p.apply(img.convert("RGB")), p


def launder_file(
    src: str | Path,
    dst: str | Path,
    profile: str,
    *,
    mask_src: str | Path | None = None,
    mask_dst: str | Path | None = None,
) -> dict:
    """Bir goruntuye profili uygular ve diske yazar.

    Maske verilirse ayni geometrik donusumler (resize/crop) maskeye de
    uygulanir ve PNG olarak kaydedilir -- maske ASLA JPEG'lenmez, cunku
    JPEG artefaktlari ikili maskeyi bozar ve piksel-F1'i sessizce dusurur.

    Doner: manifest'e yazilabilecek bilgi sozlugu.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as im:
        im = im.convert("RGB")
        orig_size = im.size
        out, p = launder_image(im, profile)

    out.save(dst, format="JPEG", quality=p.save_quality, subsampling=2)

    info = {
        "profile": profile,
        "src": str(src).replace("\\", "/"),
        "dst": str(dst).replace("\\", "/"),
        "orig_size": orig_size,
        "new_size": out.size,
        "save_quality": p.save_quality,
        "bytes": dst.stat().st_size,
        "mask_dst": "",
    }

    if mask_src and mask_dst:
        mask_out = launder_mask(mask_src, profile, target_size=out.size)
        mask_dst = Path(mask_dst)
        mask_dst.parent.mkdir(parents=True, exist_ok=True)
        mask_out.save(mask_dst, format="PNG")
        info["mask_dst"] = str(mask_dst).replace("\\", "/")

    return info


def launder_mask(
    mask_src: str | Path, profile: str, target_size: tuple[int, int]
) -> Image.Image:
    """Maskeyi goruntuyle ayni geometriye getirir.

    NOT: Kirpma yapan profiller (screenshot) icin once ayni oranla kirpip
    sonra hedef boyuta getirmek yerine dogrudan hedef boyuta NEAREST ile
    yeniden orneklemek YETERSIZDIR -- kirpma iceriği kaydirir. Bu yuzden
    kirpma adimi maskeye de aynen uygulanir.
    """
    p = LAUNDER_PROFILES[profile]
    with Image.open(mask_src) as m:
        m = m.convert("L")
        if p.crops:
            m = screenshot_crop(m)
        return m.resize(target_size, Image.NEAREST)


def binarize_mask(mask: Image.Image | np.ndarray, threshold: int = 127) -> np.ndarray:
    """Maskeyi 0/1 uint8 diziye cevirir. Piksel metrikleri her zaman ikili
    maske bekler; ara ton birakmak F1'i belirsizlestirir."""
    arr = np.asarray(mask.convert("L") if isinstance(mask, Image.Image) else mask)
    return (arr > threshold).astype(np.uint8)


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------


def _demo() -> None:
    import tempfile

    rng = np.random.default_rng(0)
    # Gercekci-ish test goruntusu: dusuk frekansli gradyan + yuksek frekansli gurultu.
    # Duz renk kullanirsak JPEG kalitesi farki byte boyutuna yansimaz ve test
    # hicbir sey dogrulamamis olur.
    h, w = 900, 1400
    yy, xx = np.mgrid[0:h, 0:w]
    base = (xx / w * 200 + yy / h * 55).astype(np.float32)
    noise = rng.normal(0, 12, (h, w))
    arr = np.clip(np.stack([base + noise, base * 0.8 + noise, base * 0.6 + noise], -1), 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))

    mask = np.zeros((h, w), np.uint8)
    mask[300:500, 400:700] = 255

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.jpg"
        msk = td / "mask.png"
        img.save(src, quality=98)
        Image.fromarray(mask).save(msk)

        print(f"{'profil':<12} {'boyut':<14} {'q':<4} {'KB':<8} maske")
        print("-" * 56)
        for name in PROFILE_NAMES:
            info = launder_file(
                src,
                td / f"out_{name}.jpg",
                name,
                mask_src=msk,
                mask_dst=td / f"mask_{name}.png",
            )
            mw, mh = Image.open(info["mask_dst"]).size
            ok = (mw, mh) == tuple(info["new_size"])
            print(
                f"{name:<12} {info['new_size']!s:<14} {info['save_quality']:<4} "
                f"{info['bytes']/1024:<8.1f} {'OK' if ok else 'UYUSMUYOR'}"
            )
            assert ok, f"{name}: maske boyutu goruntu boyutuyla uyusmuyor"

        print("\nlaunder.py sanity check OK")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Laundering profilleri")
    ap.add_argument("--demo", action="store_true", help="Sanity check calistir")
    ap.add_argument("--list", action="store_true", help="Profilleri listele")
    args = ap.parse_args()

    if args.list or not args.demo:
        for n, p in LAUNDER_PROFILES.items():
            print(f"\n[{n}]  save_quality={p.save_quality}")
            print(f"  {p.description}")
    if args.demo:
        print()
        _demo()
