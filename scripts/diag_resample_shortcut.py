"""Kalinti icerik sizintisi: forensic sinyal mi, yeniden ornekleme izi mi?

SORU
----
check_leakage.py Gorev A'da icerik ozelliklerinde AUC ~0.65-0.70 olcuyor.
Iki rakip aciklama var ve ikisi de tamamen makul:

  H1 (GERCEK SINYAL)  Difuzyon goruntuleri gercekten farkli yuksek frekans
                      istatistigi tasir (Corvi 2023, Wang 2020). Dedektorun
                      YAKALAMASI GEREKEN sey tam olarak budur; olctugumuz
                      sey bir hata degil, sinyalin kendisidir.

  H2 (KESTIRME YOL)   Havuzlarin orijinal cozunurlugu farkli oldugu icin
                      448'e inerken farkli oranda kucultuluyorlar:

                          gercek  (CarDD 1000px)  -> 448 = 0.448x
                          sd15/sd_turbo (512-768) -> 448 = 0.583-0.875x
                          sdxl    (1024-1152)     -> 448 = 0.389-0.438x

                      Az kucultulen goruntu daha cok yuksek frekans tasir.
                      Model "ne kadar yeniden orneklenmis" sorusunu ogrenir;
                      bu, kapattigimiz metadata sizintisinin piksel halidir.

Ikisi ayni tabloyu uretir. Ayirmanin tek yolu KONTROLLU KARSILASTIRMADIR.

UC TEST
-------
  TEST 1 -- ETIKETSIZ KONTROL (en belirleyici olan)
      Sadece GERCEK goruntuler. CarDD (1000px -> 0.448x) vs kendi telefon
      fotograflarimiz (4032px -> 0.111x). Ikisi de gercek; etiket farki YOK,
      yalnizca yeniden ornekleme orani farki var.
      -> AUC yuksek cikarsa ozellik kanitlanmis sekilde yeniden ornekleme
         olcuyor demektir. Bunun baska aciklamasi yok.

  TEST 2 -- ORAN ESLESTIRILMIS DILIM
      Gercek (0.448x) vs YALNIZCA sdxl (0.389-0.438x). Oranlar neredeyse
      ayni; H2 devre disi.
      -> AUC ~0.5 ise kalan sinyal yeniden ornekleme kaynakliydi.
      -> AUC yuksek kalirsa H1 dogru: gercek forensic fark var.

  TEST 3 -- ORAN UYUMSUZ DILIM
      Gercek vs sd15 + sd_turbo (0.583-0.875x). Test 2 ile arasindaki fark,
      kestirme yolun buyuklugudur.

Calistirma:
    python scripts/diag_resample_shortcut.py
    python scripts/diag_resample_shortcut.py --profile whatsapp --sample 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_leakage import content_features  # tek kaynak, kopyalama yok

SRC_MANIFEST = "data/processed/manifest_v2.parquet"
LAUNDERED = "data/processed/manifest_v2_laundered.parquet"


def attach_source_geometry(lau: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
    """Laundered satirlara TURETILDIKLERI goruntunun orijinal uzun kenarini
    ekler. Laundered satirin kendi width/height'i artik 448 -- bilgi orada
    degil, kaynakta."""
    s = src[src["launder_profile"] == "clean"].copy()
    s["src_long_edge"] = s[["width", "height"]].max(axis=1)
    key = s.set_index("source_image_id")["src_long_edge"].to_dict()
    out = lau.copy()
    out["src_long_edge"] = out["source_image_id"].map(key)
    return out


def measure(d: pd.DataFrame, y: np.ndarray, sample: int, seed: int) -> dict[str, float]:
    """Dengeli ornekle, ozellikleri cikar, AUC dondur."""
    rng = np.random.default_rng(seed)
    idx = []
    for cls in (0, 1):
        c = np.flatnonzero(y == cls)
        idx.extend(rng.choice(c, size=min(sample, len(c)), replace=False).tolist())
    idx = np.asarray(sorted(idx))

    feats, yy = [], []
    for i in idx:
        f = content_features(d.iloc[i]["path"])
        if f is None:
            continue
        feats.append(f)
        yy.append(int(y[i]))

    yy = np.asarray(yy)
    if len(feats) == 0 or yy.sum() in (0, len(yy)):
        return {}
    out = {}
    for name in feats[0]:
        x = np.asarray([f[name] for f in feats], dtype=float)
        if np.allclose(x, x[0]):
            out[name] = 0.5
            continue
        a = roc_auc_score(yy, x)
        out[name] = float(max(a, 1 - a))
    out["_n"] = float(len(yy))
    return out


def show(baslik: str, aciklama: str, aucs: dict[str, float]) -> None:
    print("-" * 70)
    print(baslik)
    print(f"  {aciklama}")
    if not aucs:
        print("  (yeterli ornek yok)")
        return
    n = int(aucs.pop("_n", 0))
    print(f"  n={n}")
    for k, v in aucs.items():
        bar = "#" * int(max(0, (v - 0.5)) * 100)
        print(f"    {k:<18} AUC {v:.3f}  {bar}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Yeniden ornekleme kestirme yolu teshisi")
    ap.add_argument("--manifest", default=LAUNDERED)
    ap.add_argument("--src-manifest", default=SRC_MANIFEST)
    ap.add_argument("--profile", default="clean")
    ap.add_argument("--sample", type=int, default=300, help="Sinif basina N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    lau = pd.read_parquet(a.manifest)
    src = pd.read_parquet(a.src_manifest)
    df = attach_source_geometry(lau, src)
    df = df[df["launder_profile"] == a.profile].reset_index(drop=True)

    print("=" * 70)
    print(f"YENIDEN ORNEKLEME TESHISI  |  profil: {a.profile}")
    print("=" * 70)
    print("\nKaynak cozunurluk -> 448 kucultme orani:")
    tab = df.groupby(["label", "generator"])["src_long_edge"].agg(["min", "max", "count"])
    tab["oran_min"] = (448 / tab["max"]).round(3)
    tab["oran_max"] = (448 / tab["min"]).round(3)
    print(tab.to_string())
    print()

    rapor: dict = {"profile": a.profile, "tests": {}}

    # --- TEST 1: etiketsiz kontrol -- gercek vs gercek --------------------
    r = df[df["label"] == "real"].reset_index(drop=True)
    y1 = (r["src_long_edge"] > 2000).astype(int).to_numpy()  # 4032'ler = 1
    if y1.sum() >= 10:
        aucs = measure(r, y1, a.sample, a.seed)
        rapor["tests"]["1_etiketsiz_kontrol"] = dict(aucs)
        show(
            "TEST 1 -- ETIKETSIZ KONTROL (ikisi de GERCEK)",
            "CarDD 1000px (0.448x)  vs  kendi fotograflarimiz 4032px (0.111x)",
            aucs,
        )
        print("  >>> Yuksek AUC = ozellik yeniden ornekleme olcuyor, KANIT.")
    else:
        print("TEST 1 atlandi: 4032px gercek ornek sayisi yetersiz.")
    print()

    # --- TEST 2: oran eslestirilmis dilim ---------------------------------
    d2 = df[
        (df["label"] == "real")
        | ((df["label"] == "fully_synthetic") & (df["src_long_edge"] >= 1024))
    ].reset_index(drop=True)
    d2 = d2[~((d2["label"] == "real") & (d2["src_long_edge"] > 2000))].reset_index(drop=True)
    y2 = (d2["label"] == "fully_synthetic").astype(int).to_numpy()
    aucs2 = measure(d2, y2, a.sample, a.seed)
    rapor["tests"]["2_oran_eslesmis"] = dict(aucs2)
    show(
        "TEST 2 -- ORAN ESLESTIRILMIS (kestirme yol devre disi)",
        "gercek CarDD 0.448x  vs  sdxl 1024/1152px 0.389-0.438x",
        aucs2,
    )
    print("  >>> Burada kalan sinyal GERCEK forensic farktir.")
    print()

    # --- TEST 3: oran uyumsuz dilim ---------------------------------------
    d3 = df[
        (df["label"] == "real")
        | ((df["label"] == "fully_synthetic") & (df["src_long_edge"] < 1024))
    ].reset_index(drop=True)
    d3 = d3[~((d3["label"] == "real") & (d3["src_long_edge"] > 2000))].reset_index(drop=True)
    y3 = (d3["label"] == "fully_synthetic").astype(int).to_numpy()
    aucs3 = measure(d3, y3, a.sample, a.seed)
    rapor["tests"]["3_oran_uyumsuz"] = dict(aucs3)
    show(
        "TEST 3 -- ORAN UYUMSUZ",
        "gercek CarDD 0.448x  vs  sd15/sd_turbo 512-768px 0.583-0.875x",
        aucs3,
    )
    print()

    # --- Yorum -------------------------------------------------------------
    print("=" * 70)
    print("YORUM")
    print("=" * 70)
    k = "dosya boyutu"
    t2 = aucs2.get(k)
    t3 = aucs3.get(k)
    if t2 is not None and t3 is not None:
        print(f"  '{k}' icin:  oran eslesmis {t2:.3f}   |   oran uyumsuz {t3:.3f}")
        fark = t3 - t2
        if fark > 0.08:
            print("  -> Kalinti sizintinin buyuk kismi YENIDEN ORNEKLEME kaynakli.")
            print("     Gorev A sonuclari oran-eslesmis dilimde AYRICA raporlanmali.")
        elif t2 > 0.65:
            print("  -> Oran eslessede sinyal duruyor: bu GERCEK forensic farktir.")
            print("     Kestirme yol degil; E3'un yakalamasi beklenen sey budur.")
        else:
            print("  -> Iki dilim de dusuk: kalinti sizinti ihmal edilebilir.")

    if a.out:
        o = Path(a.out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "results.json").write_text(
            json.dumps(rapor, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nKaydedildi: {o}/results.json")


if __name__ == "__main__":
    main()
