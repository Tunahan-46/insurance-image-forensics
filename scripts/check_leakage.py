"""Kestirme yol kapisi (plan 4.5, Tuzak 2) -- METADATA + ICERIK.

NEDEN BU DOSYA DEGISTI
----------------------
v1 yalnizca manifest'in metadata sutunlarina (width/height/en-boy/yon)
bakiyordu. Geometri normalizasyonu (src.data.launder.NORMALIZE_EDGE = 448)
devreye girdikten SONRA bu sutunlarin hepsi sabittir: her goruntu 448x448.
Yani v1 kapisi, veri ne kadar bozuk olursa olsun her zaman "YESIL" der.

Fail edemeyen bir kapi kapi degildir.

Normalizasyon metadata kanalini YAPISAL olarak kapatir; ama piksellerin
kendisinden turetilen kanallari kapatmaz. Bir goruntu 4032'den 448'e
inerken 9 kat kucultuluyorsa, 512'den 448'e inen bir goruntuye gore
belirgin sekilde daha az yuksek frekans tasir. Model forensic iz yerine
"bu goruntu ne kadar yeniden orneklenmis" sorusunu ogrenebilir ve bu
metadata sizintisinin birebir aynisidir -- sadece gorunmez halidir.

Bu yuzden kapi ikiye ayrildi:

    1. METADATA KAPISI  -- manifest sutunlari (v1'deki testler)
    2. ICERIK KAPISI    -- diskteki laundered dosyalardan olculen,
                           icerikten BAGIMSIZ olmasi gereken fiziksel
                           buyukluker

ICERIK OZELLIKLERI ve neden secildikleri
----------------------------------------
    dosya boyutu      Laundered JPEG'in diskteki boyutu. Ayni cozunurluk ve
                      ayni kalitede kaydedilmis iki goruntunun boyut farki
                      tamamen icerigin sikistirilabilirligidir. Bedava
                      hesaplanir ve olcum oncesi en guclu supheli budur.
    q95 yeniden boyut Goruntuyu sabit q95 ile yeniden sikistirinca olusan
                      boyut. Profilin kendi kalitesini denklemden cikarir;
                      boylece profiller arasi karsilastirilabilir olur.
    laplasyen var.    Yuksek frekans enerjisi. Yeniden orneklemenin en
                      dogrudan izi: cok kucultulmus goruntu duz olur.
    blokluk           JPEG 8x8 grid siniri ile grid ici komsuluklarin
                      gradyan farki. Sikistirma gecmisinin izi; splice/
                      sentetik havuzlarda farkli olabilir.
    doygunluk         HSV S ortalamasi. Bu SEMANTIK bir fark olabilir
                      (difuzyon modelleri daha canli araba uretir).
                      Kestirme yol degil dagilim kaymasidir; bu yuzden
                      INFO_ONLY -- raporlanir ama kapiyi KIRMIZI yapmaz.

YORUM (her iki kapi icin ayni)
------------------------------
    AUC ~0.50            -> sizinti yok (hedef)
    AUC 0.60-0.80        -> belirgin sizinti, duzelt
    AUC > 0.80           -> model buyuk olcude bu kanali ogreniyor
    kesisim BOS          -> KIRMIZI ALARM, %100 ayrilabilir (yalniz metadata)

Calistirma:
    python scripts/check_leakage.py
    python scripts/check_leakage.py --task A --sample 400
    python scripts/check_leakage.py --no-content          # v1 davranisi
    python scripts/check_leakage.py --out experiments/W3_leakage_gate
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_MANIFEST = "data/processed/manifest_v2_laundered.parquet"

TASKS = {
    "A": ("real", "fully_synthetic", "Gorev A -- gercek vs tam sentetik"),
    "B": ("real", "manipulated", "Gorev B -- gercek vs manipule"),
}

# Yorum esikleri
OK = 0.60
KOTU = 0.80

# Kapiyi kirmizi yapmayan, yalnizca raporlanan ozellikler (bkz. modul basligi)
INFO_ONLY = {"doygunluk"}

# Icerik ozelligi hesabinda goruntu basina okunan maksimum kenar. Laundered
# kopyalar zaten 448; buyuk bir sey gelirse (v1 manifest) hizli kalmak icin
# kucultulur -- olcumun kendisi sizinti kaynagi olmasin diye TUM goruntulere
# ayni islem uygulanir.
CONTENT_MAX_EDGE = 448


# ---------------------------------------------------------------------------
# 1. METADATA KAPISI
# ---------------------------------------------------------------------------


def geometry_features(d: pd.DataFrame) -> dict[str, np.ndarray]:
    w = d["width"].to_numpy(dtype=float)
    h = d["height"].to_numpy(dtype=float)
    return {
        "genislik": w,
        "yukseklik": h,
        "uzun kenar": np.maximum(w, h),
        "kisa kenar": np.minimum(w, h),
        "piksel sayisi": w * h,
        "en-boy": np.maximum(w, h) / np.maximum(np.minimum(w, h), 1),
        "yon (w>h)": (w > h).astype(float),
    }


# ---------------------------------------------------------------------------
# 2. ICERIK KAPISI
# ---------------------------------------------------------------------------


def _blockiness(gray: np.ndarray) -> float:
    """JPEG 8x8 grid siniri ile grid ici gecislerin gradyan farki.

    Sifira yakin deger = grid gorunmuyor. Fark buyudukce sikistirma izi
    belirginlesir. Mutlak degeri onemli degil; iki etiket arasinda FARKLI
    olmasi onemlidir."""
    dx = np.abs(np.diff(gray, axis=1))
    if dx.shape[1] < 16:
        return 0.0
    cols = np.arange(dx.shape[1])
    sinir = dx[:, cols % 8 == 7]
    ic = dx[:, cols % 8 != 7]
    return float(sinir.mean() - ic.mean())


def content_features(path: str | Path) -> dict[str, float] | None:
    """Tek bir goruntu dosyasindan icerik ozellikleri.

    Hata durumunda None doner -- eksik dosya yuzunden kapi cokmemeli, ama
    kac tanesinin okunamadigi raporlanir."""
    from PIL import Image

    p = Path(path)
    try:
        size_bytes = p.stat().st_size
        with Image.open(p) as im:
            im = im.convert("RGB")
            if max(im.size) > CONTENT_MAX_EDGE:
                s = CONTENT_MAX_EDGE / max(im.size)
                im = im.resize(
                    (max(1, round(im.width * s)), max(1, round(im.height * s))),
                    Image.LANCZOS,
                )
            arr = np.asarray(im, dtype=np.float32)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=95, subsampling=2)
            q95 = len(buf.getvalue())
            hsv = np.asarray(im.convert("HSV"), dtype=np.float32)
    except Exception:
        return None

    gray = arr.mean(axis=2)
    lap = (
        gray[2:, 1:-1] + gray[:-2, 1:-1] + gray[1:-1, 2:] + gray[1:-1, :-2]
        - 4 * gray[1:-1, 1:-1]
    )
    return {
        "dosya boyutu": float(size_bytes),
        "q95 yeniden boyut": float(q95),
        "laplasyen var.": float(lap.var()),
        "blokluk": _blockiness(gray),
        "doygunluk": float(hsv[:, :, 1].mean()),
    }


def sample_paths(
    d: pd.DataFrame, n_per_class: int, pos_label: str, seed: int
) -> pd.DataFrame:
    """Etiket basina en fazla n satir. Dengeli ornekleme; AUC'un sinif
    buyuklugunden etkilenmemesi icin degil (AUC etkilenmez) -- okuma
    maliyetini sinirlamak ve iki taraftan da yeterli ornek almak icin."""
    parts = []
    for lab in sorted(d["label"].unique()):
        sub = d[d["label"] == lab]
        if len(sub) > n_per_class:
            sub = sub.sample(n_per_class, random_state=seed)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def measure_content(
    d: pd.DataFrame, pos_label: str, n_per_class: int, seed: int
) -> tuple[dict[str, float], int, int]:
    """Ornekleme + ozellik cikarimi + AUC. Doner: (auc sozlugu, n, okunamayan)."""
    s = sample_paths(d, n_per_class, pos_label, seed)
    feats: list[dict[str, float]] = []
    y: list[int] = []
    missing = 0
    for _, row in s.iterrows():
        f = content_features(row["path"])
        if f is None:
            missing += 1
            continue
        feats.append(f)
        y.append(int(row["label"] == pos_label))

    yv = np.asarray(y)
    if len(feats) == 0 or yv.sum() == 0 or yv.sum() == len(yv):
        return {}, len(feats), missing

    aucs: dict[str, float] = {}
    for name in feats[0]:
        x = np.asarray([f[name] for f in feats], dtype=float)
        if np.allclose(x, x[0]):
            aucs[name] = 0.5
            continue
        auc = roc_auc_score(yv, x)
        aucs[name] = float(max(auc, 1 - auc))
    return aucs, len(feats), missing


# ---------------------------------------------------------------------------
# Raporlama
# ---------------------------------------------------------------------------


def _flag(auc: float, ayrik: bool = False, info: bool = False) -> str:
    if ayrik:
        return "  <<< KESISIM BOS -- %100 ayrilabilir!"
    if info:
        return "  (bilgi amacli, kapiyi etkilemez)" if auc > OK else ""
    if auc > KOTU:
        return "  <<< buyuk sizinti"
    if auc > OK:
        return "  <<  sizinti"
    return ""


def analyze_metadata(d: pd.DataFrame, pos_label: str) -> list[tuple[str, float, bool]]:
    y = (d["label"] == pos_label).astype(int).to_numpy()
    rows: list[tuple[str, float, bool]] = []
    if y.sum() == 0 or y.sum() == len(y):
        return rows

    for name, feat in geometry_features(d).items():
        if np.allclose(feat, feat[0]):
            # Butun goruntulerde ayni deger -> bilgi yok. Bu IYI bir sonuc.
            rows.append((name, 0.5, False))
            continue
        auc = roc_auc_score(y, feat)
        auc = max(auc, 1 - auc)  # yon bagimsiz: 0.2 de 0.8 kadar bilgilendirici
        pos_vals = set(np.round(feat[y == 1], 4))
        neg_vals = set(np.round(feat[y == 0], 4))
        rows.append((name, float(auc), len(pos_vals & neg_vals) == 0))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Kestirme yol kapisi (metadata + icerik)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--task", default=None, choices=list(TASKS), help="Varsayilan: hepsi")
    ap.add_argument("--profile", default=None, help="Tek bir laundering profili")
    ap.add_argument("--no-content", action="store_true", help="Yalniz metadata (v1)")
    ap.add_argument("--sample", type=int, default=250, help="Icerik: etiket basina N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="results.json yazilacak klasor")
    a = ap.parse_args()

    path = Path(a.manifest)
    if not path.exists():
        raise SystemExit(f"{path} yok.")
    df = pd.read_parquet(path)
    print(f"{path}  ({len(df)} satir)\n")

    profiles = [a.profile] if a.profile else sorted(df["launder_profile"].unique())
    tasks = [a.task] if a.task else list(TASKS)

    worst_meta = 0.0
    worst_content = 0.0
    alarm = False
    report: dict = {"manifest": str(path), "sample": a.sample, "tasks": {}}

    for t in tasks:
        neg, pos, title = TASKS[t]
        report["tasks"][t] = {}
        print("=" * 70)
        print(title)
        print("=" * 70)

        for prof in profiles:
            d = df[(df["launder_profile"] == prof) & (df["label"].isin([neg, pos]))]
            if len(d) == 0:
                continue
            cell: dict = {"n": int(len(d)), "metadata": {}, "content": {}}
            print(f"\n  [{prof}]  n={len(d)}")

            print("    -- METADATA KAPISI")
            for name, auc, ayrik in analyze_metadata(d, pos):
                worst_meta = max(worst_meta, auc)
                alarm = alarm or ayrik
                cell["metadata"][name] = {"auc": round(auc, 4), "ayrik": ayrik}
                print(f"       {name:<18} AUC {auc:.3f}{_flag(auc, ayrik)}")

            if a.no_content:
                report["tasks"][t][prof] = cell
                continue

            aucs, n_read, missing = measure_content(d, pos, a.sample, a.seed)
            print(f"    -- ICERIK KAPISI  (n={n_read}" + (f", okunamayan={missing}" if missing else "") + ")")
            if not aucs:
                print("       (yeterli ornek yok, atlandi)")
            for name, auc in aucs.items():
                info = name in INFO_ONLY
                if not info:
                    worst_content = max(worst_content, auc)
                cell["content"][name] = {"auc": round(auc, 4), "info_only": info}
                print(f"       {name:<18} AUC {auc:.3f}{_flag(auc, info=info)}")
            cell["content_n"] = n_read
            cell["content_missing"] = missing
            report["tasks"][t][prof] = cell
        print()

    print("=" * 70)
    print("SONUC")
    print("=" * 70)
    print(f"En yuksek METADATA AUC : {worst_meta:.3f}")
    if not a.no_content:
        print(f"En yuksek ICERIK   AUC : {worst_content:.3f}")
    worst = max(worst_meta, worst_content)
    report["worst_metadata_auc"] = round(worst_meta, 4)
    report["worst_content_auc"] = round(worst_content, 4)
    report["ayrik_alarm"] = alarm

    if alarm:
        durum = "KIRMIZI"
        print("DURUM: KIRMIZI -- bir metadata ozelliginde etiketler tamamen ayrik.")
        print("       Bu veriyle olculen hicbir Gorev A sonucu gecerli degil.")
    elif worst > KOTU:
        durum = "KIRMIZI"
        print("DURUM: KIRMIZI -- tek bir ozellik basli basina siniflandiriyor.")
    elif worst > OK:
        durum = "SARI"
        print("DURUM: SARI -- kalinti sizinti var, raporda belirt.")
    else:
        durum = "YESIL"
        print("DURUM: YESIL -- ne metadata ne icerik etiket hakkinda bilgi tasiyor.")
        if a.no_content:
            print("       UYARI: --no-content ile kosuldu. Bu YESIL yalnizca")
            print("       metadata icindir ve normalizasyon sonrasi bedavadir.")
    report["durum"] = durum

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nKaydedildi: {out}/results.json")

    if durum == "KIRMIZI":
        sys.exit(1)


if __name__ == "__main__":
    main()
