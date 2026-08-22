"""Senaryo basina ornek kolaj ureticisi (plan Hafta 2 Cuma teslimi).

NEDEN VAR
---------
Plan, veri setinin her senaryosundan 3 ornekli bir gorsel kolaj istiyor.
Amac ikili:
  1. dataset_card.md'ye konacak gorsel kanit
  2. GOZ DENETIMI -- uretim kalitesini sayilarla degil gozle dogrulamak.
     Kabul kapisi (changed_frac_in_mask) bir seyin degistigini soyler ama
     degisimin GERCEKCI oldugunu soylemez. Ona insan bakmali.

KVKK
----
`--include-own` verilmedikce `data/raw/own_photos` altindaki kendi telefon
fotograflarim kolaja GIRMEZ. Plakalar henuz bulaniklastirilmadi (OpenCV 5'te
CascadeClassifier kaldirildi) ve plan, sunum/paylasim gorseli uretmeden once
anonimlestirmeyi sart kosuyor. CarDD turevleri ve sentetikler bu kisitin
disinda: CarDD zaten yayinlanmis akademik bir set.

DUZEN
-----
Manipulasyon senaryolari 5 sutunlu:
    kaynak | manipule | maske bindirmesi | YAKIN kaynak | YAKIN manipule
Tam sentetik tek sutunlu:
    uretilen goruntu (kaynak fotograf yok)

NEDEN YAKIN PLAN
----------------
Maske alani goruntunun kucuk bir parcasi (inpaint_add'de medyan %2,
inpaint_enlarge'da %7). Tam kareyi 320px'e indirince degisen bolge birkac
piksele duser ve iki kare GOZLE AYNI gorunur -- oysa log'lar maske icinde
piksellerin %65-80'inin degistigini soyluyor. Yakin plan bu celiskiyi
cozer: maskenin sinirlayici kutusuna kirpip buyutur, boylece goz denetimi
gercekten yapilabilir hale gelir.

Calistirma:
    python scripts/make_collage.py                    # hepsi
    python scripts/make_collage.py --only inpaint_add # tek senaryo
    python scripts/make_collage.py --seed 7           # baska ornekler sec
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.imageio import imread, imwrite  # noqa: E402
from src.data.manifest import load_manifest  # noqa: E402

DEFAULT_MANIFEST = Path("data/processed/manifest_v2.parquet")
DEFAULT_OUT = Path("docs/assets/collages")

# Kolajdaki her hucrenin kenar uzunlugu. 320: A4'e basildiginda okunur,
# repo'ya commit edilecek kadar kucuk (senaryo basina ~200 KB).
CELL = 320
PAD = 8
LABEL_H = 26

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (250, 250, 250)
FG = (30, 30, 30)
MASK_TINT = (0, 0, 255)  # BGR -- maske bindirmesi kirmizi


def _fit(img: np.ndarray, size: int = CELL) -> np.ndarray:
    """Goruntuyu kareye oturtur: en-boy oranini KORUR, bosluga BG doldurur.

    Kirpma yerine doldurma tercih edildi -- kirpmak hasari kadraj disinda
    birakabilir ve kolajin amaci tam da hasari gostermek."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4
    small = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((size, size, 3), BG, np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2
    canvas[y:y + nh, x:x + nw] = small
    return canvas


def _overlay_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Maskeyi yari saydam kirmizi olarak bindirir + konturunu cizer.

    Sadece tint yeterli degil: acik renkli araclarda %35 alfa zor secilir.
    Kontur, maskenin sinirini net gosterir -- maske kalitesini (dikdortgen mi,
    duzensiz mi; yumusak kenarli mi) gozle denetlemeyi mumkun kilar."""
    if mask is None:
        return img
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    binm = (mask > 127).astype(np.uint8)
    out = img.copy()
    tint = np.full_like(img, MASK_TINT)
    idx = binm.astype(bool)
    out[idx] = cv2.addWeighted(img, 0.65, tint, 0.35, 0)[idx]
    contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, MASK_TINT, 2)
    return out


def _mask_bbox(mask: np.ndarray, shape, margin: float = 0.6, min_frac: float = 0.18):
    """Maskenin sinirlayici kutusunu, cevresinde pay birakarak doner.

    Pay sart: hasarin kendisi kadar, cevresindeki DOKUNULMAMIS yuzeyle
    olan gecisi de gormek gerekiyor -- yamanin belli olup olmadigi tam
    orada anlasilir. min_frac ise cok kucuk maskelerde asiri yakinlasip
    piksel bulamacina donmeyi engeller.
    """
    H, W = shape[:2]
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1

    side = int(max(bw, bh) * (1 + margin))
    side = max(side, int(min(H, W) * min_frac))
    side = min(side, min(H, W))

    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x = int(np.clip(cx - side // 2, 0, W - side))
    y = int(np.clip(cy - side // 2, 0, H - side))
    return x, y, side


def _crop(img: np.ndarray, box) -> np.ndarray:
    x, y, s = box
    return img[y:y + s, x:x + s]


def _label_strip(text: str, width: int) -> np.ndarray:
    strip = np.full((LABEL_H, width, 3), BG, np.uint8)
    cv2.putText(strip, text[:60], (4, 18), FONT, 0.45, FG, 1, cv2.LINE_AA)
    return strip


def _row(cells: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """Etiketli hucreleri yatay birlestirir."""
    blocks = []
    for title, img in cells:
        block = np.vstack([_label_strip(title, CELL), _fit(img)])
        blocks.append(block)
        blocks.append(np.full((block.shape[0], PAD, 3), BG, np.uint8))
    return np.hstack(blocks[:-1]) if blocks else np.zeros((1, 1, 3), np.uint8)


def _source_path(gen_params: str) -> str | None:
    """gen_params JSON'undan kaynak fotograf yolunu cikarir."""
    import json

    try:
        d = json.loads(gen_params) if isinstance(gen_params, str) else dict(gen_params or {})
    except Exception:
        return None
    return d.get("source_path") or d.get("original_path")


def build_scenario(rows, title: str, out_path: Path) -> bool:
    """Tek bir senaryo icin kolaj uretir. Basarili olursa True doner."""
    strips: list[np.ndarray] = []

    for _, row in rows.iterrows():
        img = imread(row["path"])
        if img is None:
            continue

        if row["label"] == "fully_synthetic":
            gp = row.get("gen_params") or ""
            import json

            try:
                d = json.loads(gp) if isinstance(gp, str) else dict(gp)
            except Exception:
                d = {}
            cells = [(f"{d.get('model', row['generator'])}  {img.shape[1]}x{img.shape[0]}", img)]
        else:
            cells = []
            src_img = None
            src = _source_path(row.get("gen_params") or "")
            if src and Path(src).exists():
                src_img = imread(src)
                if src_img is not None:
                    cells.append(("kaynak (gercek)", src_img))
            cells.append((f"manipule ({row['generator']})", img))

            mp = str(row.get("mask_path") or "")
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if mp and Path(mp).exists() else None
            if m is not None:
                cells.append(("maske bindirmesi", _overlay_mask(img, m)))
                box = _mask_bbox(m, img.shape)
                if box is not None:
                    zoom = 100 * box[2] / max(img.shape[:2])
                    if src_img is not None and src_img.shape[:2] == img.shape[:2]:
                        cells.append((f"YAKIN kaynak  (%{zoom:.0f})", _crop(src_img, box)))
                    cells.append((f"YAKIN manipule (%{zoom:.0f})", _crop(img, box)))

        strips.append(_row(cells))

    if not strips:
        print(f"  {title}: gosterilecek ornek bulunamadi, atlandi")
        return False

    width = max(s.shape[1] for s in strips)
    padded = []
    for s in strips:
        if s.shape[1] < width:
            s = np.hstack([s, np.full((s.shape[0], width - s.shape[1], 3), BG, np.uint8)])
        padded.append(s)
        padded.append(np.full((PAD, width, 3), BG, np.uint8))

    header = np.full((34, width, 3), BG, np.uint8)
    cv2.putText(header, title, (6, 24), FONT, 0.7, FG, 2, cv2.LINE_AA)

    collage = np.vstack([header] + padded[:-1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(out_path, collage, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  {title:<20} -> {out_path}  ({collage.shape[1]}x{collage.shape[0]})")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Senaryo basina ornek kolaj uret")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n", type=int, default=3, help="Senaryo basina ornek sayisi")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="Sadece bu manip_type / 'synthetic'")
    ap.add_argument("--include-own", action="store_true",
                    help="KVKK: kendi telefon fotograflarini da dahil et (plakalar bulanik DEGIL)")
    a = ap.parse_args()

    df = load_manifest(a.manifest)
    # Yalnizca ham (laundered olmayan) satirlar -- kolajin amaci uretim
    # kalitesini gostermek, sikistirma bozulmasini degil.
    df = df[df["launder_profile"].astype(str) == "clean"]

    if not a.include_own:
        n_before = len(df)
        df = df[~df["path"].astype(str).str.contains("own_photos")]
        dropped = n_before - len(df)
        if dropped:
            print(f"KVKK: {dropped} kendi fotografim kolaj disi birakildi "
                  f"(--include-own ile dahil edilebilir).")

    out_dir = Path(a.out)
    rng = np.random.default_rng(a.seed)
    made = 0

    scenarios: list[tuple[str, object]] = []
    synth = df[df["label"] == "fully_synthetic"]
    if len(synth):
        scenarios.append(("fully_synthetic", synth))
    for mt in sorted(df[df["label"] == "manipulated"]["manip_type"].unique()):
        scenarios.append((mt, df[df["manip_type"] == mt]))

    for name, pool in scenarios:
        if a.only and a.only != name:
            continue
        if len(pool) == 0:
            continue
        # Cesitlilik: ayni kaynak fotograftan iki ornek secilmesin
        pool = pool.drop_duplicates(subset=["source_image_id"])
        k = min(a.n, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        made += build_scenario(pool.iloc[idx], name, out_dir / f"{name}.jpg")

    print(f"\n{made} kolaj uretildi -> {out_dir}")
    print("Bunlari docs/dataset_card.md'ye referansla ve GOZLE denetle:")
    print("  - maskeler duzensiz ve yumusak kenarli mi (dikdortgen DEGIL)?")
    print("  - eklenen hasar gercekci mi, yoksa 'yamali' mi duruyor?")
    print("  - silinen hasar temiz mi, hayalet iz birakmis mi?")


if __name__ == "__main__":
    main()
