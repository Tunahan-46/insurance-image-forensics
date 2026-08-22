"""M3 (klasik manipulasyon) katmanini Kaggle icin ZIP'e paketler.

NEDEN AYRI BIR SCRIPT
----------------------
`data/raw/manipulated/classic` yerelde OpenCV ile uretildi (splice,
copy_move, bg_replace) -- diffusers/GPU gerektirmedigi icin W2'nin Kaggle
zip'lerine (w2_synthetic.zip, w2_manipulated.zip = S + M1 + M2) hic
GIRMEDI. Notebook'un 7. hucresi ucuncu bir zip bekliyor: w2_classic.zip.

PowerShell'in `Compress-Archive`'i kullanilmiyor -- ZIP icinde ters bolu
(\\) yazar, Kaggle "yasaklanmis karakter" diyip yuklemeyi reddeder
(bkz. scripts/make_data_zip.py basligi, ayni tuzak orada da belgelendi).
Python zipfile POSIX ayirici ('/') kullanir, platformdan bagimsizdir.

DUZ YAPI -- ONEMLI
------------------
notebooks/W3_kaggle_clip_embed.ipynb hucre 7 zip'i DOGRUDAN
`data/raw/manipulated/classic`'e ac (`z.extractall(dst)`, ek kok klasor
YOK). Bu yuzden zip icindeki yollar `cardd_xxx_splice.png`,
`masks/cardd_xxx_splice_mask.png` seklinde DUZ olmali -- CarDD zip'lerindeki
gibi bir "classic/" onegiyle SARILMAMALI. Bu script bunu garanti eder.

Calistirma (proje kokunde):
    python scripts/make_classic_zip.py
Cikti:
    w2_classic.zip   (repo kokunde, Kaggle'a Input olarak yuklenecek)
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

SRC = Path("data/raw/manipulated/classic")
OUT = Path("w2_classic.zip")


def verify(out_path: Path) -> bool:
    """Zip icinde ters bolu KALMADIGINI dogrular -- Kaggle yuklemesinin
    sonunda degil, burada, saniyeler icinde."""
    with zipfile.ZipFile(out_path) as z:
        names = z.namelist()
    bad = [n for n in names if "\\" in n]
    if bad:
        print(f"  DOGRULAMA BASARISIZ: {len(bad)} yolda ters bolu var.")
        print(f"    ornek: {bad[0]}")
        return False
    print(f"  dogrulama: TEMIZ ({len(names)} yol, hepsi '/' ayiricili)")
    print(f"    ornek: {names[0]}")
    return True


def main() -> None:
    if not SRC.exists():
        print(f"HATA: {SRC} yok. Once M3'un uretildiginden emin ol.")
        sys.exit(1)

    files = [p for p in sorted(SRC.rglob("*")) if p.is_file()]
    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"{OUT.name}: {len(files)} dosya, {total_mb:.0f} MB")

    if OUT.exists():
        print(f"{OUT.name} zaten var, siliniyor (yeniden uretilecek).")
        OUT.unlink()

    t0 = time.time()
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for i, p in enumerate(files, 1):
            # as_posix(): ayirici HER ZAMAN "/" -- Windows'ta bile.
            arc = p.relative_to(SRC).as_posix()
            z.write(p, arcname=arc)
            if i % 100 == 0:
                print(f"  {i}/{len(files)}  ({time.time()-t0:.0f} sn)")

    size_mb = OUT.stat().st_size / 1e6
    print(f"  tamam: {OUT}  ({size_mb:.0f} MB, {time.time()-t0:.0f} sn)")

    if not verify(OUT):
        sys.exit(1)
    print(f"\nHazir. Kaggle'daki datasetine (input) {OUT.name} dosyasini ekle.")


if __name__ == "__main__":
    main()
