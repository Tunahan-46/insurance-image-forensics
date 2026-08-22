"""Splice/copy_move harmanlama teshisi.

SORU
----
Kolajlarda splice orneklerinde renk uyumsuzlugu goze carpti. `_paste_region`
splice'ta %70 olasilikla cv2.seamlessClone (Poisson) kullaniyor olmali;
ama `except cv2.error: pass` SESSIZ, yani seamlessClone atarsa kod fark
ettirmeden alfa harmanlamaya duser ve bunu hicbir yere yazmaz.

Bu script log'a bakmaz -- URETILMIS GORUNTULERI olcer.

NASIL
-----
Poisson blending kaynagin mutlak rengini atar, yalnizca gradyanini tasir ve
sinirdan hedefin rengini iceri yayar. Yani:

    Poisson calistiysa : maske icindeki ortalama renk, maskenin HEMEN
                         DISINDAKI halkaya yakin olmali
    Alfa'ya dustuyse   : maske ici, cevresinden bagimsiz kalir; donorun
                         rengi aynen durur

Olcut: maske icindeki ortalama Lab rengi ile maskeyi ceviren halkanin
ortalama Lab rengi arasindaki mesafe (yaklasik dE). Karsilastirma tabani
olarak AYNI olcum kaynak (manipule edilmemis) goruntude de yapilir --
boylece "bu araba zaten iki renkliydi" durumu ayirt edilir.

Calistirma:
    python scripts/diag_splice.py
    python scripts/diag_splice.py --type copy_move --n 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.imageio import imread  # noqa: E402

LOG = Path("data/raw/manipulated/classic/gen_log.jsonl")

# Halka genisligi: maske sinirinin hemen disi. Cok dar olursa Poisson'un
# zaten esitledigi sinir pikselleri olcume girer ve her sey uyumlu gorunur;
# cok genis olursa arka plan/zemin karisir.
RING_INNER = 6
RING_OUTER = 26

# dE esikleri (CIE76 kabaca): 2.3 gozle ayirt edilebilir sinir,
# 10+ "belirgin farkli renk", 25+ "alakasiz renk".
BELIRGIN = 10.0
ALAKASIZ = 25.0


def _lab_mean(img_bgr: np.ndarray, m: np.ndarray) -> np.ndarray | None:
    if m.sum() == 0:
        return None
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV 8-bit Lab olcegi: L 0-255, a/b 0-255 (128 merkezli).
    # dE icin gercek olcege cevir.
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0
    return lab[m.astype(bool)].reshape(-1, 3).mean(axis=0)


def _ring(mask: np.ndarray) -> np.ndarray:
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RING_INNER * 2 + 1,) * 2)
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RING_OUTER * 2 + 1,) * 2)
    binm = (mask > 127).astype(np.uint8)
    return (cv2.dilate(binm, k_out) - cv2.dilate(binm, k_in)).clip(0, 1)


def analyze(rec: dict) -> dict | None:
    manip = imread(rec["path"])
    src = imread(rec["source_path"])
    mask = cv2.imread(rec["mask_path"], cv2.IMREAD_GRAYSCALE)
    if manip is None or src is None or mask is None:
        return None
    if src.shape[:2] != manip.shape[:2]:
        src = cv2.resize(src, (manip.shape[1], manip.shape[0]))
    if mask.shape[:2] != manip.shape[:2]:
        mask = cv2.resize(mask, (manip.shape[1], manip.shape[0]), interpolation=cv2.INTER_NEAREST)

    inner = (mask > 127).astype(np.uint8)
    ring = _ring(mask)
    if inner.sum() < 50 or ring.sum() < 50:
        return None

    a = _lab_mean(manip, inner)
    b = _lab_mean(manip, ring)
    a0 = _lab_mean(src, inner)
    b0 = _lab_mean(src, ring)
    if a is None or b is None or a0 is None or b0 is None:
        return None

    return {
        "path": rec["path"],
        "manip_type": rec.get("manip_type", "?"),
        # manipule goruntude: maske ici <-> cevresi
        "de_manip": float(np.linalg.norm(a - b)),
        # kaynak goruntude ayni olcum -- dogal renk farki tabani
        "de_kaynak": float(np.linalg.norm(a0 - b0)),
        "mask_area_frac": rec.get("mask_area_frac"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Splice/copy_move harmanlama teshisi")
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("--type", default=None, help="splice / copy_move / bg_replace")
    ap.add_argument("--n", type=int, default=0, help="0 = hepsi")
    a = ap.parse_args()

    log = Path(a.log)
    if not log.exists():
        raise SystemExit(f"{log} yok. classic_manip calistirilmis mi?")

    recs = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.type:
        recs = [r for r in recs if r.get("manip_type") == a.type]
    if a.n:
        recs = recs[: a.n]
    print(f"{len(recs)} kayit inceleniyor...\n")

    rows = [r for r in (analyze(rec) for rec in recs) if r]
    if not rows:
        raise SystemExit("Hicbir kayit olculemedi (dosyalar eksik olabilir).")

    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["manip_type"], []).append(r)

    print(f"{'tip':<14}{'n':>5}{'dE manip':>12}{'dE kaynak':>12}{'artis':>9}"
          f"{'>10':>7}{'>25':>7}")
    print("-" * 68)
    for mt, rs in sorted(by.items()):
        dm = np.array([r["de_manip"] for r in rs])
        dk = np.array([r["de_kaynak"] for r in rs])
        n_bel = int((dm > BELIRGIN).sum())
        n_ala = int((dm > ALAKASIZ).sum())
        print(f"{mt:<14}{len(rs):>5}{np.median(dm):>12.1f}{np.median(dk):>12.1f}"
              f"{np.median(dm) - np.median(dk):>9.1f}"
              f"{100*n_bel/len(rs):>6.0f}%{100*n_ala/len(rs):>6.0f}%")

    print()
    print("dE manip  : manipule goruntude maske ici <-> cevre renk mesafesi")
    print("dE kaynak : AYNI olcum kaynak fotografta (dogal fark tabani)")
    print("artis     : manipulasyonun getirdigi ek renk kopuklugu")
    print(">10 / >25 : belirgin / alakasiz renk farki olan orneklerin orani")
    print()
    print("YORUM:")
    print("  artis ~0        -> harmanlama calisiyor, renk kopuklugu yok")
    print("  artis 5-15      -> kismi; bazi ornekler alfa'ya dusmus olabilir")
    print("  artis 15+       -> harmanlama buyuk olcude devre disi")

    worst = sorted(rows, key=lambda r: -r["de_manip"])[:10]
    print("\nEn kotu 10 ornek (gozle bak):")
    for r in worst:
        print(f"  dE={r['de_manip']:6.1f} (kaynak {r['de_kaynak']:5.1f})  "
              f"alan={r['mask_area_frac']}  {Path(r['path']).name}")

    out = Path("experiments/diag_splice.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nAyrinti: {out}")


if __name__ == "__main__":
    main()
