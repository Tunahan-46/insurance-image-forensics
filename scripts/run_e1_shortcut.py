"""
E1 -- KESTIRME YOL (SHORTCUT) TESHIS DENEYI.

Plan 4.5, Tuzak 2 ve 3'un sonundaki emir:
    "Eger AUC'un 0.99 cikiyorsa sevinme -- once bu iki tuzagi kontrol et.
     Hizli bir teshis testi: goruntuleri 32x32'ye kucultup ayni modeli
     egit. Hala %95 aliyorsan model forensic iz degil, dusuk seviyeli
     istatistiksel bir kestirme yol ogreniyor demektir. Bu testi mutlaka
     yap ve sonucunu raporla."

W1 BULGUSU (bu deneyin sebebi)
------------------------------
E0'da egitilmemis ResNet-50 ile AUC 0.364 cikti -- tesadufun ALTINDA.
Egitilmemis bir model sistematik olarak ters yonde ayiriyorsa, iki sinif
arasinda dusuk seviyeli bir istatistiksel fark var demektir. Hipotez:
cozunurluk (CarDD ~1000px, sentetikler 512px) ve JPEG kalitesi.

Bu script hipotezi OLCER, tahmin etmez.

DORT PROB (artan zorluk sirasi)
--------------------------------
    P0  meta       : SADECE genislik, yukseklik, en-boy orani, dosya boyutu,
                     piksel basina byte. HIC PIKSEL OKUMAZ.
                     -> Yuksek AUC = felaket. Model resmin icine bakmadan
                        siniflandirabiliyor demektir.
    P1  px8        : 8x8 gri piksel (64 ozellik). Neredeyse tum uzamsal
                     bilgi yok edilmis; geriye kaba renk/parlaklik kaliyor.
    P2  px32       : 32x32 gri piksel -- PLANIN ISTEDIGI TEST.
    P3  px32_rgb   : 32x32 renkli (3072 ozellik). Renk dagilimi kestirmesi.

Her prob icin siniflandirici: L2-regularize lojistik regresyon.
Neden derin model degil: kestirme yol testinin amaci EN ZAYIF modelin bile
ayirt edip edemedigini gormek. Zayif model ayirt ediyorsa sinyal veri
setinde, modelde degil.

DEGERLENDIRME
-------------
Her prob x her laundering profili icin ayri ROC-AUC. Cunku hipotezin
ikinci yarisi su: laundering (ozellikle `aggressive`, ki hepsini 1024px'e
indirir ve q60 yapar) cozunurluk/kalite kestirmesini KAPATMALI. Eger
`clean`de AUC 0.95 ve `aggressive`de 0.55 ise, kestirme yol dogrulanmis
olur ve laundering'in bunu kapattigi gosterilmis olur -- bu, sunumun
en guclu slaytlarindan biri.

Iki gorev ayri ayri olculur:
    A  real  vs  fully_synthetic   (Task A)
    B  real  vs  manipulated       (Task B'nin detection kismi)

Calistirma:
    python scripts/run_e1_shortcut.py
    python scripts/run_e1_shortcut.py --manifest data/processed/manifest_v2_laundered.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_MANIFEST = "data/processed/manifest_v2_laundered.parquet"
OUT_DIR = Path("experiments/E1_shortcut")

PROBES = ("meta", "px8", "px32", "px32_rgb")
TASKS = {
    "A_synthetic": ("real", "fully_synthetic"),
    "B_manipulated": ("real", "manipulated"),
}

# Karar esikleri -- yorumu okuyucuya birakmamak icin.
AUC_ALARM = 0.75  # bunun ustu: ciddi kestirme yol
AUC_WATCH = 0.65  # bunun ustu: dikkat


# Ozellik onbellegi: path -> {probe: vektor}.
#
# NEDEN: ayni goruntu 4 prob x (birden fazla profil kombinasyonu) icin
# tekrar tekrar decode ediliyordu. JPEG decode bu scriptin suresinin
# %95'idir; problar ise ayni decode'dan turetilebilir. Tek acilista
# hepsini cikarmak calisma suresini ~4 kat kisaltir.
_CACHE: dict[str, dict[str, np.ndarray] | None] = {}


def features_all_probes(path: str) -> dict[str, np.ndarray] | None:
    """Bir goruntuyu BIR KEZ acar, dort probun ozelliklerini birden dondurur."""
    if path in _CACHE:
        return _CACHE[path]
    p = Path(path)
    try:
        nbytes = p.stat().st_size
        with Image.open(p) as im:
            w, h = im.size
            # HIC PIKSEL OKUNMUYOR: sadece dosya ve boyut istatistigi.
            meta = np.array([
                w, h, max(w, h), min(w, h), w * h,
                w / h, nbytes, nbytes / (w * h),
            ], dtype=np.float64)
            rgb = im.convert("RGB")
            gray = rgb.convert("L")
            out = {
                "meta": meta,
                "px8": np.asarray(gray.resize((8, 8), Image.BILINEAR),
                                  dtype=np.float64).reshape(-1) / 255.0,
                "px32": np.asarray(gray.resize((32, 32), Image.BILINEAR),
                                   dtype=np.float64).reshape(-1) / 255.0,
                "px32_rgb": np.asarray(rgb.resize((32, 32), Image.BILINEAR),
                                       dtype=np.float64).reshape(-1) / 255.0,
            }
    except Exception:
        out = None
    _CACHE[path] = out
    return out


def extract_features(path: str, probe: str) -> np.ndarray | None:
    f = features_all_probes(path)
    return None if f is None else f[probe]


def build_xy(df: pd.DataFrame, probe: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, y, ids = [], [], []
    for _, r in df.iterrows():
        f = extract_features(str(r["path"]), probe)
        if f is None:
            continue
        X.append(f)
        y.append(int(r["_pos"]))
        ids.append(str(r["source_image_id"]))
    if not X:
        return np.empty((0, 0)), np.empty(0), []
    return np.vstack(X), np.array(y), ids


def fit_and_score(
    train_df: pd.DataFrame, test_df: pd.DataFrame, probe: str
) -> tuple[float | None, int, int]:
    Xtr, ytr, _ = build_xy(train_df, probe)
    Xte, yte, _ = build_xy(test_df, probe)
    if len(Xtr) == 0 or len(Xte) == 0:
        return None, len(Xtr), len(Xte)
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None, len(Xtr), len(Xte)

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    s = clf.predict_proba(sc.transform(Xte))[:, 1]
    return float(roc_auc_score(yte, s)), len(Xtr), len(Xte)


def verdict(auc: float | None) -> str:
    if auc is None:
        return "-"
    if auc >= AUC_ALARM:
        return "ALARM"
    if auc >= AUC_WATCH:
        return "dikkat"
    return "ok"


def run_task(df: pd.DataFrame, task: str, profiles: list[str]) -> dict:
    neg_label, pos_label = TASKS[task]
    sub = df[df["label"].isin([neg_label, pos_label])].copy()
    sub["_pos"] = (sub["label"] == pos_label).astype(int)

    print("\n" + "=" * 78)
    print(f"GOREV {task}:  {neg_label}  vs  {pos_label}")
    print("=" * 78)

    if len(sub) == 0:
        print("  Bu gorev icin veri yok, atlaniyor.")
        return {}

    n_pos = int(sub["_pos"].sum())
    if n_pos == 0:
        print(f"  '{pos_label}' katmani manifest'te BOS (0 ornek) -- gorev atlaniyor.")
        print("  Bu bir hata degil: S katmani Colab uretimi bekliyor.")
        return {}

    # Egitim: train split, KARISIK profiller (augmentation gibi).
    train_all = sub[sub["split"] == "train"]

    header = f"{'prob':<12}" + "".join(f"{p:<14}" for p in profiles)
    print(header)
    print("-" * len(header))

    results: dict[str, dict] = {}
    unmatched: set[str] = set()
    for probe in PROBES:
        row = {}
        cells = ""
        for prof in profiles:
            te = sub[(sub["split"] == "test") & (sub["launder_profile"] == prof)]
            # Egitimi ayni profille sinirlamak, "profil x profil" degil
            # "profil icinde" bir teshis verir -- kestirme yolun o profilde
            # HALA var olup olmadigini sorar. Dogru soru budur.
            tr = train_all[train_all["launder_profile"] == prof]
            matched = len(tr) >= 8
            if not matched:
                # O profil train'de uretilmedi (screenshot/aggressive
                # train/val'de uretilmiyor). Karisik profille egitmek
                # ZORUNDA kaliyoruz -- ama bu artik "profil icinde
                # kestirme yol var mi" sorusunu degil, "farkli profilde
                # egitilmis model bu profile genellenebiliyor mu"
                # sorusunu olcer. Sonuc * ile isaretlenir; yorum motoru
                # bu hucreleri kanit saymaz.
                tr = train_all
                unmatched.add(prof)
            auc, ntr, nte = fit_and_score(tr, te, probe)
            row[prof] = {
                "auc": auc,
                "n_train": ntr,
                "n_test": nte,
                "train_profile_matched": matched,
            }
            txt = f"{auc:.3f}{'' if matched else '*'} {verdict(auc)}" if auc is not None else "-"
            cells += f"{txt:<14}"
        results[probe] = row
        print(f"{probe:<12}{cells}")

    if unmatched:
        print(f"\n  * = bu profil train split'inde uretilmemis ({sorted(unmatched)});")
        print("    model karisik profillerle egitildi. Bu hucreler PROFIL-ICI")
        print("    kestirme yol kaniti DEGILDIR -- dagilim kaymasi da ayni")
        print("    dususu uretebilir. Kesin olcum icin:")
        print(f"    python scripts/apply_laundering.py --profiles {' '.join(sorted(unmatched))} --splits train")

    return results


def interpret(results: dict) -> list[str]:
    """Sayilari yoruma cevir. Rapor bunu aynen W2.md'ye tasir."""
    notes: list[str] = []
    for task, probes in results.items():
        if not probes:
            continue
        meta = probes.get("meta", {})
        clean_meta = (meta.get("clean") or {}).get("auc")
        px32 = probes.get("px32", {})
        clean_px32 = (px32.get("clean") or {}).get("auc")
        agg_px32 = (px32.get("aggressive") or {}).get("auc")

        if clean_meta is not None and clean_meta >= AUC_ALARM:
            notes.append(
                f"[{task}] META prob AUC={clean_meta:.3f}: model TEK BIR PIKSEL "
                f"OKUMADAN ayirt edebiliyor. Cozunurluk/dosya boyutu dagilimlari "
                f"sinifa gore farkli. Plan 4.5 Tuzak 2/3 DOGRULANDI."
            )
        if clean_px32 is not None and clean_px32 >= AUC_ALARM:
            notes.append(
                f"[{task}] 32x32 prob AUC={clean_px32:.3f} (>= {AUC_ALARM}): "
                f"planin teshis testi POZITIF. Yuksek cozunurlukte alacagin "
                f"her AUC degeri bu kadarlik bir kestirme yol ICERIR."
            )
        agg_matched = (px32.get("aggressive") or {}).get("train_profile_matched", True)
        if clean_px32 is not None and agg_px32 is not None:
            drop = clean_px32 - agg_px32
            if drop >= 0.10 and not agg_matched:
                notes.append(
                    f"[{task}] 32x32 AUC clean={clean_px32:.3f} -> "
                    f"aggressive={agg_px32:.3f} (dusus {drop:.3f}) AMA aggressive "
                    f"profili train'de uretilmedigi icin model karisik profille "
                    f"egitildi. Bu dusus kestirme yolun kapandigini DEGIL, "
                    f"train/test dagilim kaymasini da yansitiyor olabilir. "
                    f"Iddiada bulunmadan once aggressive profilini train'de de "
                    f"uret ve tekrar olc."
                )
            elif drop >= 0.10:
                notes.append(
                    f"[{task}] Laundering kestirme yolu KAPATIYOR: 32x32 AUC "
                    f"clean={clean_px32:.3f} -> aggressive={agg_px32:.3f} "
                    f"(dusus {drop:.3f}, ikisi de profil-ici egitim). Bu, "
                    f"laundering katmaninin sadece 'gercekcilik' degil, "
                    f"BILIMSEL GECERLILIK icin de gerekli oldugunun kaniti."
                )
            elif drop <= -0.10:
                notes.append(
                    f"[{task}] BEKLENMEDIK: aggressive profilde AUC clean'den "
                    f"YUKSEK ({agg_px32:.3f} > {clean_px32:.3f}). Laundering "
                    f"yeni bir kestirme yol YARATMIS olabilir -- profil "
                    f"parametrelerini gozden gecir."
                )
        if clean_px32 is not None and clean_px32 < AUC_WATCH:
            notes.append(
                f"[{task}] 32x32 AUC={clean_px32:.3f} < {AUC_WATCH}: dusuk "
                f"seviyeli kestirme yol ZAYIF. Veri seti bu acidan saglikli; "
                f"Hafta 3'te alacagin AUC gercek sinyal sayilabilir."
            )
    return notes


def main() -> None:
    ap = argparse.ArgumentParser(description="E1: kestirme yol teshisi")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--profiles", nargs="*", default=None)
    a = ap.parse_args()

    mp = Path(a.manifest)
    if not mp.exists():
        print(f"HATA: {mp} yok.")
        print("Once: python scripts/build_manifest_v2.py && python scripts/apply_laundering.py")
        sys.exit(1)

    df = pd.read_parquet(mp) if mp.suffix == ".parquet" else pd.read_csv(mp)
    profiles = a.profiles or sorted(
        df[df["split"] == "test"]["launder_profile"].unique().tolist()
    )

    print("=" * 78)
    print("E1 -- KESTIRME YOL TESHISI (plan 4.5, Tuzak 2 ve 3)")
    print("=" * 78)
    print(f"Manifest : {mp}  ({len(df)} satir)")
    print(f"Profiller: {profiles}")
    print("\nProblar  : meta=sadece boyut/dosya-byte (piksel OKUMAZ)")
    print("           px8/px32/px32_rgb = kucultulmus ham piksel")
    print(f"Esikler  : AUC >= {AUC_ALARM} ALARM, >= {AUC_WATCH} dikkat")

    results = {t: run_task(df, t, profiles) for t in TASKS}

    print("\n" + "=" * 78)
    print("YORUM")
    print("=" * 78)
    notes = interpret(results)
    if not notes:
        print("  Otomatik yorum uretilemedi (yeterli veri yok).")
    for n in notes:
        print(f"\n  * {n}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": str(mp),
        "profiles": profiles,
        "thresholds": {"alarm": AUC_ALARM, "watch": AUC_WATCH},
        "results": results,
        "notes": notes,
    }
    (out / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Markdown tablo -- dogrudan W2.md'ye yapistirilabilir.
    lines = ["| gorev | prob | " + " | ".join(profiles) + " |",
             "|---|---|" + "---|" * len(profiles)]
    starred = False
    for task, probes in results.items():
        for probe, row in probes.items():
            cells = []
            for p in profiles:
                cell = row.get(p) or {}
                v = cell.get("auc")
                if v is None:
                    cells.append("-")
                    continue
                mark = "" if cell.get("train_profile_matched", True) else "*"
                starred = starred or bool(mark)
                cells.append(f"{v:.3f}{mark}")
            lines.append(f"| {task} | {probe} | " + " | ".join(cells) + " |")
    if starred:
        lines.append("")
        lines.append("`*` = bu profil train split'inde uretilmedigi icin model karisik "
                     "profillerle egitildi; hucre profil-ici kestirme yol kaniti degildir.")
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nKaydedildi:\n  {out}/results.json\n  {out}/table.md")
    print("\nBu tabloyu docs/weekly/W2.md'ye yapistir. Sonucu ne olursa olsun")
    print("RAPORLA -- plan 4.5 acikca 'bu testi mutlaka yap ve sonucunu")
    print("raporla' diyor. Negatif sonuc da bir bulgudur.")


if __name__ == "__main__":
    main()
