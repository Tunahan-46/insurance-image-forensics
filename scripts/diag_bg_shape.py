"""bg_replace icin ikinci tur teshis.

SORU
----
diag_splice.py dE fix'ten SONRA da bg_replace'te degisim gostermedi
(>25 orani %39 -> %36, artis 4.2 -> 5.6). Bu BEKLENEN OLABILIR: bg_replace
icin renk kapisi bilerek uygulanmadi (arka plan zaten farkli bir sahne).
Ama en kotu 10 ornekte artis 40-90 araligina cikan asiri sapmalar var
(orn. cardd_003640: kaynak dE=3.0, manip dE=91.7). Bu, sadece "dogal
sahne farki" ile aciklanamayacak kadar buyuk.

Iki olasi kok neden:
  (a) shape_is_rectangular esigi (0.92) COK GEVSEK -- gercekte bozuk olan
      segmentasyonlar 0.85-0.92 araliginda kalip kapiyi gecebiliyor.
  (b) segmentasyon dogru (dikdortgen degil) ama GaussianBlur(21,21) genis
      kontrastli sinirda yetersiz -- bu FARKLI bir sorun, esik ayari
      degil, harmanlama genisligi meselesi.

Bu script experiments/diag_splice.json (diag_splice.py'nin cikitisi) ile
gercek maske dosyalarini birlestirip, dE'ye gore siralanmis bir tabloda
'extent' (kontur alani / sinirlayici dikdortgen alani) sutunu ekler.
Yuksek dE + yuksek extent (>0.85) -> (a); yuksek dE + dusuk extent -> (b).

Calistirma:
    python scripts/diag_bg_shape.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def extent_of(mask_path: Path, invert: bool = False) -> float | None:
    """Maskenin disini alan konturunun 'extent' (dikdortgene yakinlik) degeri.

    invert=True: bg_replace icin ZEMIN GERCEGI maskesi ARKA PLAN'dir
    (NOT(arac govdesi)) -- ceperi resmin cercevesine yapisik "cerceve"
    seklindedir, RETR_EXTERNAL bu durumda ic deligi (araci) GORMEZ ve
    kontur alani HER ZAMAN ~tum resim olur (extent daima ~1.0). Uretim
    sirasinda gercekte kontrol edilen sey `body` (arac govdesi, ters
    CEVRILMEDEN ONCEKI hali) oldugundan, ayni seyi olcmek icin burada
    maskeyi TERSINE cevirip (yani govdeyi geri elde edip) olcuyoruz.
    """
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if invert:
        m = cv2.bitwise_not(m)
    contours, _ = cv2.findContours(
        (m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    _, _, w, h = cv2.boundingRect(c)
    if w * h == 0:
        return None
    return float(area / (w * h))


def main() -> None:
    diag_path = Path("experiments/diag_splice.json")
    if not diag_path.exists():
        raise SystemExit(f"{diag_path} yok. Once diag_splice.py calistir.")

    rows = json.loads(diag_path.read_text(encoding="utf-8"))
    rows = [r for r in rows if r.get("manip_type") == "bg_replace"]
    if not rows:
        raise SystemExit("diag_splice.json icinde bg_replace kaydi yok.")

    for r in rows:
        p = Path(r["path"])
        mask_path = p.parent / "masks" / p.name
        r["extent"] = extent_of(mask_path, invert=True)

    rows = [r for r in rows if r["extent"] is not None]
    rows.sort(key=lambda r: -r["de_manip"])

    print(f"{'dE manip':>10}{'dE kaynak':>11}{'artis':>8}{'extent':>9}{'alan':>8}  dosya")
    print("-" * 80)
    for r in rows[:25]:
        artis = r["de_manip"] - r["de_kaynak"]
        print(f"{r['de_manip']:>10.1f}{r['de_kaynak']:>11.1f}{artis:>8.1f}"
              f"{r['extent']:>9.3f}{r['mask_area_frac']:>8.3f}  {Path(r['path']).name}")

    extents = np.array([r["extent"] for r in rows])
    des = np.array([r["de_manip"] - r["de_kaynak"] for r in rows])
    print()
    print(f"extent > 0.92 (mevcut esik) : {int((extents > 0.92).sum())}/{len(rows)}")
    print(f"extent > 0.85               : {int((extents > 0.85).sum())}/{len(rows)}")
    print(f"extent > 0.80               : {int((extents > 0.80).sum())}/{len(rows)}")
    # korelasyon: yuksek artis'li orneklerde extent de yuksek mi?
    top20_idx = np.argsort(-des)[:20]
    print(f"\nEn yuksek 'artis'li 20 ornegin ortalama extent'i: "
          f"{extents[top20_idx].mean():.3f} (genelin ortalamasi: {extents.mean():.3f})")


if __name__ == "__main__":
    main()
