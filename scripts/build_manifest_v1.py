"""
Hafta 1 manifest olusturucu (CarDD gercek yapisina gore ayarlanmis).

KESIF SONUCLARI (inspect_cardd.py ciktisindan):
  - CarDD_COCO: 4000 goruntu (train2017=2816, val2017=810, test2017=374)
  - CarDD_SOD : AYNI 4000 goruntu + maskeler + edge haritalari
                (CarDD-TR/VAL/TE altinda -Image, -Mask, -Edge klasorleri)
  => Goruntu kaynagi olarak SADECE CarDD_COCO kullanilir.
     Ikisini birden okumak ayni goruntuyu iki kez manifeste sokar ve
     split sizintisina yol acar (plan 4.5, Tuzak 1).

  - CarDD'nin KENDI train/val/test bolumu var. Kendi hash'imizi
    uydurmak yerine bunu kullaniyoruz: kuratorlu bir bolum ve
    SOD maske klasorleriyle birebir eslesiyor.

  - 4000 adet piksel-seviyesi HASAR maskesi mevcut. Bunlari Hafta 2'de
    inpainting bolgesi olarak kullanacagiz (SAM adimini atlatir).
    DIKKAT: Bu maskeler HASAR maskesidir, MANIPULASYON maskesi degil.
    Bu yuzden manifest'in mask_path alanina YAZILMAZ (o alan Hafta 2'de
    uretilecek manipulasyon maskeleri icin ayrilmistir). Simdilik
    gen_params icinde referans olarak saklaniyor.

Calistirma:
    python scripts/build_manifest_v1.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest import (  # noqa: E402
    add_row,
    check_split_leakage,
    new_manifest,
    save_manifest,
)

# ---------------------------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------------------------

CARDD_ROOT = Path("data/raw/cardd")
CARDD_COCO = CARDD_ROOT / "CarDD_COCO"
CARDD_SOD = CARDD_ROOT / "CarDD_SOD"

# COCO klasor adi -> bizim split adimiz
COCO_SPLIT_MAP = {
    "train2017": "train",
    "val2017": "val",
    "test2017": "test",
}

# Bizim split adimiz -> SOD maske klasoru
SOD_MASK_DIRS = {
    "train": CARDD_SOD / "CarDD-TR" / "CarDD-TR-Mask",
    "val": CARDD_SOD / "CarDD-VAL" / "CarDD-VAL-Mask",
    "test": CARDD_SOD / "CarDD-TE" / "CarDD-TE-Mask",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Kendi fotograflarin ve sentetikler icin hash tabanli split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Hafta 1: sentetiklerin tamami test'e (bkz. main() icindeki aciklama).
# Hafta 2'de veri seti buyuyunce False yap.
SYNTH_ALL_TO_TEST = True

OUTPUT_PATH = "data/processed/manifest_v1.parquet"

# ---------------------------------------------------------------------------


def deterministic_split(source_image_id: str) -> str:
    """CarDD disi kaynaklar icin deterministik split."""
    h = int(hashlib.md5(source_image_id.encode()).hexdigest(), 16)
    r = (h % 10000) / 10000.0
    if r < TRAIN_RATIO:
        return "train"
    if r < TRAIN_RATIO + VAL_RATIO:
        return "val"
    return "test"


def safe_image_size(path: Path):
    try:
        with Image.open(path) as img:
            return img.size
    except Exception as e:
        print(f"  [atlandi] {path.name}: {e}")
        return None


def collect_images(root: Path):
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def find_damage_mask(stem: str, split: str) -> str:
    """SOD klasorlerinde ayni isimli hasar maskesini bul."""
    mask_dir = SOD_MASK_DIRS.get(split)
    if mask_dir is None or not mask_dir.exists():
        return ""
    candidate = mask_dir / f"{stem}.png"
    if candidate.exists():
        return str(candidate).replace("\\", "/")
    return ""


def add_cardd(df):
    """CarDD_COCO'dan gercek goruntuleri ekle, kendi split'ini kullan."""
    print(f"[1/4] CarDD taraniyor: {CARDD_COCO}")
    if not CARDD_COCO.exists():
        print(f"  UYARI: {CARDD_COCO} bulunamadi, atlaniyor.")
        return df, 0, 0

    added = 0
    skipped = 0
    masks_found = 0

    for coco_dir, split in COCO_SPLIT_MAP.items():
        folder = CARDD_COCO / coco_dir
        if not folder.exists():
            print(f"  UYARI: {folder} yok, atlaniyor.")
            continue

        images = collect_images(folder)
        print(f"  {coco_dir}/ -> split='{split}': {len(images)} goruntu")

        for p in images:
            size = safe_image_size(p)
            if size is None:
                skipped += 1
                continue

            sid = f"cardd_{p.stem}"
            damage_mask = find_damage_mask(p.stem, split)
            if damage_mask:
                masks_found += 1

            df = add_row(
                df,
                source_image_id=sid,
                path=str(p).replace("\\", "/"),
                label="real",
                width=size[0],
                height=size[1],
                split=split,
                launder_profile="clean",
                gen_params={"damage_mask_path": damage_mask} if damage_mask else {},
            )
            added += 1

    print(f"  Toplam eklendi: {added}")
    print(f"  Hasar maskesi eslesen: {masks_found}/{added}")
    if added > 0 and masks_found < added * 0.9:
        print("  UYARI: Maskelerin cogu eslesmedi. SOD_MASK_DIRS yollarini kontrol et.")
    return df, added, skipped


def add_simple_source(df, root: Path, prefix: str, label: str,
                      generator: str = "none", force_test_ratio=None,
                      force_split=None):
    """own_photos / roboflow / synthetic_quick icin ortak ekleme mantigi."""
    images = collect_images(root)
    if not images:
        return df, 0, 0

    added = 0
    skipped = 0
    for i, p in enumerate(images):
        size = safe_image_size(p)
        if size is None:
            skipped += 1
            continue
        sid = f"{prefix}_{p.stem}"
        if force_split:
            split = force_split
        elif force_test_ratio:
            split = "test" if i % force_test_ratio != 0 else "train"
        else:
            split = deterministic_split(sid)

        df = add_row(
            df,
            source_image_id=sid,
            path=str(p).replace("\\", "/"),
            label=label,
            generator=generator,
            width=size[0],
            height=size[1],
            split=split,
            launder_profile="clean",
        )
        added += 1
    return df, added, skipped


def main() -> None:
    df = new_manifest()
    stats = {"cardd": 0, "own": 0, "roboflow": 0, "synthetic": 0, "skipped": 0}

    df, n, s = add_cardd(df)
    stats["cardd"] = n
    stats["skipped"] += s

    print("\n[2/4] Kendi fotograflarin: data/raw/own_photos")
    df, n, s = add_simple_source(
        df, Path("data/raw/own_photos"), "own", "real", force_test_ratio=5
    )
    stats["own"] = n
    stats["skipped"] += s
    if n == 0:
        print("  Henuz fotograf yok. Otoparkta 50-100 fotograf cekmeyi unutma!")
        print("  (Kablo/AirDrop ile aktar - WhatsApp EXIF'i siler)")
    else:
        print(f"  {n} goruntu eklendi")

    print("\n[3/4] Roboflow: data/raw/roboflow")
    df, n, s = add_simple_source(df, Path("data/raw/roboflow"), "robo", "real")
    stats["roboflow"] = n
    stats["skipped"] += s
    print(f"  {n} goruntu eklendi" if n else "  Yok (opsiyonel, sorun degil)")

    print("\n[4/4] Sentetik: data/raw/synthetic_quick")
    # HAFTA 1 KARARI: sentetiklerin TAMAMI test'e konur.
    # Gerekce: elimizde sadece ~20 sentetik var. %70/15/15 bolersek test'e
    # 3 tane duser ve 374 real'e karsi 3 fake ile hesaplanan hicbir metrik
    # anlamli olmaz. E0'in amaci zaten metrik degil, zincir testi.
    # Hafta 2'de 1200+ sentetik uretilince SYNTH_ALL_TO_TEST=False yapilip
    # duzgun oranlara gecilecek.
    df, n, s = add_simple_source(
        df, Path("data/raw/synthetic_quick"), "synth", "fully_synthetic",
        generator="sd15",
        force_split="test" if SYNTH_ALL_TO_TEST else None,
    )
    stats["synthetic"] = n
    stats["skipped"] += s
    if n == 0:
        print("  Henuz sentetik goruntu yok.")
        print("  Colab'da 20 adet uretmeyi unutma (README bolum 1.4)")
    else:
        print(f"  {n} goruntu eklendi")

    print("\n" + "=" * 60)
    print("DOGRULAMA")
    print("=" * 60)

    if len(df) == 0:
        print("HATA: Hicbir goruntu bulunamadi. Klasor yollarini kontrol et.")
        sys.exit(1)

    problems = check_split_leakage(df)
    if problems:
        print("SIZINTI TESPIT EDILDI - manifest KAYDEDILMEDI:")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    print("Split sizintisi kontrolu: TEMIZ")

    test_labels = set(df[df["split"] == "test"]["label"].unique())
    print(f"Test setindeki etiketler: {test_labels}")
    if len(test_labels) < 2:
        print("\n  >>> UYARI: Test setinde tek sinif var (hepsi 'real').")
        print("  >>> Bu NORMAL - henuz sentetik goruntu uretmedin.")
        print("  >>> E0'in metrik kismi calismayacak, ama manifest gecerli.")
        print("  >>> Colab'da 20 sentetik uretip bu script'i tekrar calistir.")

    print("\n" + "=" * 60)
    print("OZET")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  TOPLAM: {len(df)} satir")

    print("\nSplit x Label dagilimi:")
    print(df.groupby(["split", "label"]).size().to_string())

    n_with_mask = df["gen_params"].str.contains("damage_mask_path").sum()
    print(f"\nHasar maskesi referansi olan satir: {n_with_mask}")

    save_manifest(df, OUTPUT_PATH)
    print("\nTamam. Sonraki adim: python scripts/run_e0.py")


if __name__ == "__main__":
    main()