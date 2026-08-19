"""CarDD verisini Colab/Kaggle icin ZIP'e paketler.

NEDEN AYRI BIR SCRIPT
---------------------
PowerShell'in `Compress-Archive` komutu, ZIP icindeki dosya yollarini
Windows ters bolusuyle yazar:

    CarDD_COCO\\test2017\\000012.jpg      <- BOZUK

ZIP spesifikasyonu (APPNOTE 4.4.17) yol ayiricinin ileri bolu olmasini
sart kosar. Colab'in Python `zipfile` modulu buna toleranslidir ve sorunsuz
acar; Kaggle ise katidir ve "yasaklanmis karakter" diyerek yuklemeyi
REDDEDER. Ayni zip iki platformda farkli davranir.

Python `zipfile` her zaman POSIX ayirici yazar. Bu script bu yuzden var:
platformdan bagimsiz, her yerde acilan arsiv uretir.

SIKISTIRMA: ZIP_STORED (sikistirma yok) kullanilir. Icerik zaten JPEG/PNG,
yani onceden sikistirilmis. Deflate uygulamak dakikalarca CPU yakar ve
boyutu ~%1-2 dusurur -- kotu bir takas.

Calistirma:
    python scripts/make_data_zip.py                 # ikisini de uret
    python scripts/make_data_zip.py --what masks    # sadece SOD maskeleri
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

CARDD = Path("data/raw/cardd")

# Kaynak klasor -> zip icindeki kok ad.
# SOD'un SADECE maske klasorleri alinir: CarDD_SOD icinde goruntulerin bir
# kopyasi daha var (CarDD-*-Image, ~3 GB) ve build_manifest_v2.py onlara
# hic bakmaz -- yalnizca CarDD-*-Mask okunur.
TARGETS: dict[str, list[tuple[Path, str]]] = {
    "coco": [(CARDD / "CarDD_COCO", "CarDD_COCO")],
    "masks": [
        (CARDD / "CarDD_SOD/CarDD-TR/CarDD-TR-Mask", "CarDD-TR-Mask"),
        (CARDD / "CarDD_SOD/CarDD-VAL/CarDD-VAL-Mask", "CarDD-VAL-Mask"),
        (CARDD / "CarDD_SOD/CarDD-TE/CarDD-TE-Mask", "CarDD-TE-Mask"),
    ],
}

OUT_NAMES = {"coco": "cardd_coco.zip", "masks": "cardd_sod_masks.zip"}


def build_zip(out_path: Path, sources: list[tuple[Path, str]]) -> None:
    missing = [str(s) for s, _ in sources if not s.exists()]
    if missing:
        print(f"HATA: kaynak klasor(ler) yok:\n  " + "\n  ".join(missing))
        sys.exit(1)

    files: list[tuple[Path, str]] = []
    for src, root in sources:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                # as_posix(): ayirici HER ZAMAN "/" olur.
                rel = p.relative_to(src).as_posix()
                files.append((p, f"{root}/{rel}"))

    total_mb = sum(p.stat().st_size for p, _ in files) / 1e6
    print(f"{out_path.name}: {len(files)} dosya, {total_mb:.0f} MB")

    t0 = time.time()
    with zipfile.ZipFile(
        out_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as z:
        for i, (p, arc) in enumerate(files, 1):
            z.write(p, arcname=arc)
            if i % 500 == 0:
                print(f"  {i}/{len(files)}  ({time.time()-t0:.0f} sn)")

    size_mb = out_path.stat().st_size / 1e6
    print(f"  tamam: {out_path}  ({size_mb:.0f} MB, {time.time()-t0:.0f} sn)")


def verify(out_path: Path) -> bool:
    """Zip icinde ters bolu KALMADIGINI dogrular.

    Bu kontrol olmadan hata ancak Kaggle yuklemesinin sonunda ortaya cikar
    -- 3 GB'i bosuna yukledikten sonra."""
    with zipfile.ZipFile(out_path) as z:
        names = z.namelist()
    bad = [n for n in names if "\\" in n]
    if bad:
        print(f"  DOGRULAMA BASARISIZ: {len(bad)} yolda ters bolu var.")
        print(f"    ornek: {bad[0]}")
        return False
    print(f"  dogrulama: TEMIZ ({len(names)} yol, hepsi '/' ayiricili)")
    print(f"    ornek: {names[min(1, len(names)-1)]}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="CarDD verisini ZIP'e paketle")
    ap.add_argument("--what", choices=["coco", "masks", "all"], default="all")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["coco", "masks"] if a.what == "all" else [a.what]

    ok = True
    for key in keys:
        out_path = out_dir / OUT_NAMES[key]
        if out_path.exists():
            print(f"{out_path.name} zaten var, siliniyor (yeniden uretilecek).")
            out_path.unlink()
        build_zip(out_path, TARGETS[key])
        ok &= verify(out_path)
        print()

    if not ok:
        sys.exit(1)
    print("Hepsi hazir. Kaggle'a bu dosyalari yukle.")


if __name__ == "__main__":
    main()
