"""
CarDD klasor yapisini kesfet.

Amac: manifest yazmadan ONCE veri setinde tam olarak ne oldugunu gormek.
  - Hangi klasorde kac goruntu var
  - COCO ve SOD ayni goruntuleri mi iceriyor (duplikasyon riski)
  - Maske dosyalari nerede (Hafta 2'de inpainting icin lazim olacak)

Calistirma:
    python scripts/inspect_cardd.py
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

CARDD_ROOT = Path("data/raw/cardd")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def print_tree(root: Path, max_depth: int = 3, prefix: str = "") -> None:
    """Klasor agacini max_depth seviyeye kadar yazdir."""
    if max_depth < 0:
        return
    try:
        entries = sorted([p for p in root.iterdir() if p.is_dir()])
    except (PermissionError, FileNotFoundError):
        return
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "`-- " if is_last else "|-- "
        n_images = len([p for p in entry.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
        print(f"{prefix}{connector}{entry.name}/  [{n_images} goruntu]")
        extension = "    " if is_last else "|   "
        print_tree(entry, max_depth - 1, prefix + extension)


def count_by_folder(root: Path) -> dict[str, int]:
    """Her dogrudan alt klasorde kac goruntu var (recursive)."""
    counts = {}
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            counts[sub.name] = len(
                [p for p in sub.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
            )
    return counts


def file_hash(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def check_duplicates(root: Path, sample_limit: int = 400) -> None:
    """COCO ve SOD ayni goruntuleri mi iceriyor?

    Once dosya ADI ile hizli kontrol, sonra ornek bir alt kume uzerinde
    ICERIK hash'i ile kesin kontrol. Icerik hash'i yavas oldugu icin
    sadece ilk sample_limit dosyada yapilir.
    """
    print("\n" + "=" * 70)
    print("DUPLIKASYON KONTROLU (COCO vs SOD)")
    print("=" * 70)

    all_images = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    print(f"Toplam goruntu dosyasi (tum klasorler): {len(all_images)}")

    # 1) Dosya adi bazli kontrol
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for p in all_images:
        by_stem[p.stem].append(p)

    repeated = {k: v for k, v in by_stem.items() if len(v) > 1}
    print(f"Birden fazla yerde gecen dosya ADI sayisi: {len(repeated)}")

    if repeated:
        print("\nOrnek tekrarlar (ilk 3):")
        for stem, paths in list(repeated.items())[:3]:
            print(f"  '{stem}' -> {len(paths)} kez:")
            for p in paths[:4]:
                print(f"      {p.relative_to(root)}")

    # 2) Icerik hash'i ile kesin kontrol (ornek uzerinde)
    print(f"\nIcerik hash kontrolu (ilk {sample_limit} dosya)...")
    sample = all_images[:sample_limit]
    hashes: dict[str, list[Path]] = defaultdict(list)
    for p in sample:
        try:
            hashes[file_hash(p)].append(p)
        except OSError:
            continue

    identical = {h: ps for h, ps in hashes.items() if len(ps) > 1}
    n_unique = len(hashes)
    print(f"  Incelenen: {len(sample)} dosya")
    print(f"  Benzersiz icerik: {n_unique}")
    print(f"  Ayni icerige sahip grup sayisi: {len(identical)}")

    if identical:
        print("\n  >>> UYARI: Ayni goruntu birden fazla yerde bulunuyor.")
        print("  >>> Manifest'e SADECE BIR klasoru kaynak olarak ekle,")
        print("  >>> aksi halde split sizintisi olusur (plan 4.5, Tuzak 1).")
        for h, ps in list(identical.items())[:2]:
            print(f"    Ayni icerik:")
            for p in ps[:4]:
                print(f"      {p.relative_to(root)}")
    else:
        print("  Ornek kumede duplikasyon bulunamadi.")


def find_masks(root: Path) -> None:
    """SOD formati piksel-seviyesi maske icerir. Bu maskeler Hafta 2'de
    inpainting bolgesi secmek icin ALTIN DEGERINDE - SAM ile maske
    uretmeye gerek kalmadan hazir hasar maskelerin olur."""
    print("\n" + "=" * 70)
    print("MASKE / ANOTASYON DOSYALARI")
    print("=" * 70)

    mask_keywords = ["mask", "gt", "label", "annotation", "seg"]
    mask_dirs = []
    for p in root.rglob("*"):
        if p.is_dir() and any(k in p.name.lower() for k in mask_keywords):
            n = len([q for q in p.rglob("*") if q.suffix.lower() in IMAGE_EXTS])
            mask_dirs.append((p.relative_to(root), n))

    if mask_dirs:
        print("Maske icerebilecek klasorler:")
        for rel, n in sorted(mask_dirs):
            print(f"  {rel}/  [{n} dosya]")
        print("\n  >>> NOT: Bu maskeleri Hafta 2'de inpainting bolgesi olarak")
        print("  >>> kullanabilirsin. SAM ile maske uretme adimini atlatir.")
    else:
        print("Isimden maske klasoru tespit edilemedi.")

    json_files = list(root.rglob("*.json"))
    if json_files:
        print(f"\nJSON anotasyon dosyalari ({len(json_files)} adet):")
        for j in json_files[:10]:
            size_mb = j.stat().st_size / (1024 * 1024)
            print(f"  {j.relative_to(root)}  ({size_mb:.1f} MB)")


def main() -> None:
    if not CARDD_ROOT.exists():
        print(f"HATA: {CARDD_ROOT} bulunamadi.")
        print("CarDD arsivini data/raw/cardd/ altina cikardigindan emin ol.")
        return

    print("=" * 70)
    print(f"KLASOR AGACI: {CARDD_ROOT}")
    print("=" * 70)
    print(f"{CARDD_ROOT.name}/")
    print_tree(CARDD_ROOT, max_depth=3)

    print("\n" + "=" * 70)
    print("UST SEVIYE KLASORLERDE GORUNTU SAYISI")
    print("=" * 70)
    for name, n in count_by_folder(CARDD_ROOT).items():
        print(f"  {name}: {n}")

    ext_counter = Counter(
        p.suffix.lower() for p in CARDD_ROOT.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )
    print("\nDosya uzantisi dagilimi:")
    for ext, n in ext_counter.most_common():
        print(f"  {ext}: {n}")

    check_duplicates(CARDD_ROOT)
    find_masks(CARDD_ROOT)

    print("\n" + "=" * 70)
    print("SONRAKI ADIM")
    print("=" * 70)
    print("Bu ciktiyi mentoruna gonder. Ozellikle sunlar onemli:")
    print("  1. Hangi klasorde kac goruntu var")
    print("  2. Duplikasyon var mi (COCO vs SOD ayni goruntuler mi)")
    print("  3. Maske klasorleri nerede")
    print("Buna gore scripts/build_manifest_v1.py ayarlanacak.")


if __name__ == "__main__":
    main()
