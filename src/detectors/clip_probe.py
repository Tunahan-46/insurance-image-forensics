"""E3 -- dondurulmus CLIP + lineer prob (plan Hafta 3, ANA BASELINE).

Ojha vd. (CVPR 2023) paradigmasi: ozellik uzayina DOKUNMA, uzerine tek bir
lineer sinir koy. Gerekce (bkz. docs/lit/01_ojha_universal_fake_detect.md):
uctan uca egitilen dedektorler egitildikleri generator'un artefaktina
asimetrik olarak kilitlenir ve gorulmemis generator'i "gercek" tarafina iter.
Dondurulmus CLIP uzayinda bu asimetri olusmuyor.

Bizim icin ikinci bir gerekce daha var: S katmanimiz 330 goruntu. Uctan uca
bir ViT'i bu veriyle fine-tune etmek asiri uyumdan baska bir sey uretmez.
Lineer prob tam da bu rejim icin.

RECETE (plan, 5 adim)
---------------------
  1. CLIP ViT-L/14 dondur                    -> src/features/clip_embed.py
  2. 768-d embedding cikar, .npy cache'le    -> src/features/clip_embed.py
  3. LogisticRegression(class_weight='balanced') + grid search   [bu dosya]
  4. VAL setinde threshold sec (TPR@FPR=1% maksimize)            [bu dosya]
  5. Platt scaling ile kalibre et            -> src/eval/calibration.py

KRITIK KURALLAR
---------------
* Grid search ve threshold secimi YALNIZCA val uzerinde. Test setine
  bakarak C secmek plan 4.5 Tuzak 5'tir; test seti donduruldu ve
  sha256'lendi (bkz. docs/dataset_card.md).
* Egitim ve degerlendirme YALNIZCA laundered kopyalar uzerinden. Ham
  dosyalar farkli formatlarda (PNG vs JPEG) -- format kestirme yolu.
* Gorev A'da (real vs fully_synthetic) cozunurluk kestirme yolu ACIK
  (W3 Bulgu 9-10, meta prob AUC ~0.98). Bu modelin Gorev A'daki AUC'u
  duzeltme yapilmadan forensic sinyal olarak YORUMLANAMAZ. Gorev B'de
  boyle bir kisit yok.

Calistirma:
    python -m src.detectors.clip_probe --task A --cache data/processed/clip_cache
    python -m src.detectors.clip_probe --task B --profiles clean whatsapp
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_CACHE = Path("data/processed/clip_cache")
DEFAULT_MANIFEST = Path("data/processed/manifest_v2_laundered.parquet")

# Plan 8.3 hedefleri TPR@FPR=%1 uzerinden konusmuyor ama triage senaryosunda
# yanlis alarm pahalidir: her yanlis alarm bir eksperin masasina dosya koyar.
TARGET_FPR = 0.01

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


# ---------------------------------------------------------------------------
# Gorev tanimlari
# ---------------------------------------------------------------------------

TASKS = {
    "A": {"positive": ["fully_synthetic"], "negative": ["real"], "name": "A_synthetic"},
    "B": {"positive": ["manipulated"], "negative": ["real"], "name": "B_manipulated"},
    "AB": {
        "positive": ["fully_synthetic", "manipulated"],
        "negative": ["real"],
        "name": "AB_any_fake",
    },
}


def task_labels(df, task: str) -> np.ndarray:
    """Etiket sutunundan ikili hedef uretir. Ilgisiz satirlar -1 alir."""
    spec = TASKS[task]
    lab = df["label"].astype(str)
    y = np.full(len(df), -1, dtype=int)
    y[lab.isin(spec["negative"]).to_numpy()] = 0
    y[lab.isin(spec["positive"]).to_numpy()] = 1
    return y


# ---------------------------------------------------------------------------
# Esik secimi
# ---------------------------------------------------------------------------

def threshold_at_fpr(scores: np.ndarray, y: np.ndarray, target_fpr: float = TARGET_FPR) -> dict:
    """Verilen FPR butcesinde TPR'i maksimize eden esigi bulur.

    ROC uzerinde FPR <= hedef olan en yuksek TPR noktasi secilir. Hicbir
    nokta butceye sigmiyorsa (kucuk negatif sinif) en dusuk FPR'li nokta
    dondurulur ve gercek FPR raporlanir -- sessizce butceyi asmaz."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, scores)
    ok = fpr <= target_fpr + 1e-12
    if ok.any():
        i = int(np.argmax(np.where(ok, tpr, -1)))
    else:
        i = int(np.argmin(fpr))
    return {
        "threshold": float(thr[i]),
        "tpr": float(tpr[i]),
        "fpr": float(fpr[i]),
        "target_fpr": target_fpr,
        "budget_met": bool(fpr[i] <= target_fpr + 1e-12),
    }


# ---------------------------------------------------------------------------
# Dedektor
# ---------------------------------------------------------------------------

@dataclass
class CLIPProbe:
    """Dondurulmus CLIP embedding'leri uzerinde lineer prob.

    Girdi olarak GORUNTU degil EMBEDDING alir -- embedding cikarma pahali ve
    bir kereliktir (bkz. src/features/clip_embed.py). Bu ayrim sayesinde
    grid search / threshold / kalibrasyon dakikalar icinde doner.
    """

    C: float = 1.0
    task: str = "A"
    clf: object = None
    calibrator: object = None
    threshold: float = 0.5
    meta: dict = field(default_factory=dict)

    # -- egitim ------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CLIPProbe":
        from sklearn.linear_model import LogisticRegression

        self.clf = LogisticRegression(
            C=self.C,
            class_weight="balanced",  # gercek 4052 / sahte 1170 (~3.5:1)
            max_iter=5000,
            solver="lbfgs",
        )
        self.clf.fit(X, y)
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        """Ham (kalibre edilmemis) karar skoru. AUC bunun uzerinden."""
        if self.clf is None:
            raise RuntimeError("Once fit() cagir.")
        return self.clf.decision_function(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Kalibre edilmis olasilik. Kalibrator yoksa lojistik cikti."""
        s = self.decision(X)
        if self.calibrator is not None:
            return self.calibrator.transform(s)
        return 1.0 / (1.0 + np.exp(-s))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision(X) >= self.threshold).astype(int)

    # -- kalici hale getirme ----------------------------------------------

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "clf": self.clf,
                "calibrator": self.calibrator,
                "threshold": self.threshold,
                "C": self.C,
                "task": self.task,
                "meta": self.meta,
            },
            path,
        )

    @staticmethod
    def load(path: str | Path) -> "CLIPProbe":
        import joblib

        d = joblib.load(path)
        p = CLIPProbe(C=d["C"], task=d["task"])
        p.clf = d["clf"]
        p.calibrator = d["calibrator"]
        p.threshold = d["threshold"]
        p.meta = d.get("meta", {})
        return p


# ---------------------------------------------------------------------------
# Egitim akisi
# ---------------------------------------------------------------------------

def train_probe(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    *,
    task: str = "A",
    c_grid: list[float] | None = None,
    calibration: str = "platt",
    target_fpr: float = TARGET_FPR,
    verbose: bool = True,
) -> tuple[CLIPProbe, dict]:
    """Grid search + esik secimi + kalibrasyon. Hepsi VAL uzerinde.

    Model secim olcutu ROC-AUC degil **TPR@FPR=1%**: triage senaryosunda
    yanlis alarm dogrudan is yuku demek. Iki C ayni AUC'u verip cok farkli
    dusuk-FPR davranisi gosterebilir."""
    from sklearn.metrics import roc_auc_score

    from src.eval.calibration import fit_calibrator

    c_grid = c_grid or C_GRID
    results = []

    for C in c_grid:
        p = CLIPProbe(C=C, task=task).fit(X_tr, y_tr)
        s_va = p.decision(X_va)
        auc = float(roc_auc_score(y_va, s_va))
        thr = threshold_at_fpr(s_va, y_va, target_fpr)
        results.append({"C": C, "val_auc": auc, **thr})
        if verbose:
            flag = "" if thr["budget_met"] else "  (FPR butcesi tutmadi)"
            print(f"  C={C:<8g} val AUC={auc:.4f}  TPR@FPR{target_fpr:.0%}={thr['tpr']:.4f}{flag}")

    best = max(results, key=lambda r: (r["tpr"], r["val_auc"]))
    if verbose:
        print(f"  -> secilen C={best['C']:g}  (TPR={best['tpr']:.4f}, AUC={best['val_auc']:.4f})")

    probe = CLIPProbe(C=best["C"], task=task).fit(X_tr, y_tr)
    probe.threshold = best["threshold"]

    s_va = probe.decision(X_va)
    probe.calibrator = fit_calibrator(s_va, y_va, kind=calibration)
    probe.meta = {
        "grid": results,
        "selected": best,
        "calibration": calibration,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "train_pos_rate": float(np.mean(y_tr)),
    }
    return probe, probe.meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sklearn.metrics import roc_auc_score

    from src.data.manifest import load_manifest
    from src.features.clip_embed import align_to_manifest, load_embeddings

    ap = argparse.ArgumentParser(description="E3: dondurulmus CLIP + lineer prob")
    ap.add_argument("--task", default="A", choices=list(TASKS))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--cache", nargs="+", default=[str(DEFAULT_CACHE)],
                    help="Bir veya birden fazla embedding cache klasoru")
    ap.add_argument("--out", default=None, help="experiments/E3_clip_probe gibi")
    ap.add_argument("--profiles", nargs="*", default=None,
                    help="Egitimde kullanilacak laundering profilleri (varsayilan: hepsi)")
    ap.add_argument("--calibration", default="platt", choices=["platt", "isotonic"])
    ap.add_argument("--target-fpr", type=float, default=TARGET_FPR)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out_dir = Path(a.out or f"experiments/E3_clip_probe_{TASKS[a.task]['name']}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"E3 -- CLIP lineer prob | gorev {a.task} ({TASKS[a.task]['name']})")
    print("=" * 70)

    df = load_manifest(a.manifest)
    X, ids = load_embeddings(a.cache)
    X, df = align_to_manifest(X, ids, df)

    y = task_labels(df, a.task)
    keep = y >= 0
    X, df, y = X[keep], df[keep], y[keep]

    if a.profiles:
        m = df["launder_profile"].astype(str).isin(a.profiles).to_numpy()
        # Filtre yalnizca TRAIN/VAL icin; test her profilde ayri raporlanir
        m |= (df["split"].astype(str) == "test").to_numpy()
        X, df, y = X[m], df[m], y[m]

    split = df["split"].astype(str).to_numpy()
    tr, va, te = split == "train", split == "val", split == "test"
    print(f"train {tr.sum()} | val {va.sum()} | test {te.sum()}")
    print(f"pozitif orani: train {y[tr].mean():.3f}  val {y[va].mean():.3f}")
    print()

    probe, meta = train_probe(
        X[tr], y[tr], X[va], y[va],
        task=a.task, calibration=a.calibration, target_fpr=a.target_fpr,
    )

    # --- TEST: profil basina AYRI rapor (plan sarti) ----------------------
    print("\nTEST -- laundering profili basina")
    print("-" * 70)
    per_profile = {}
    prof = df["launder_profile"].astype(str).to_numpy()
    for p in sorted(set(prof[te])):
        m = te & (prof == p)
        if m.sum() == 0 or len(set(y[m])) < 2:
            print(f"  {p:<14} atlandi (tek sinif veya bos)")
            continue
        s = probe.decision(X[m])
        auc = float(roc_auc_score(y[m], s))
        thr = threshold_at_fpr(s, y[m], a.target_fpr)
        pred = (s >= probe.threshold).astype(int)
        acc = float((pred == y[m]).mean())
        per_profile[p] = {
            "n": int(m.sum()), "auc": auc, "acc_at_val_threshold": acc,
            "tpr_at_fpr_in_profile": thr["tpr"],
        }
        print(f"  {p:<14} n={m.sum():<6} AUC={auc:.4f}  "
              f"acc@val-esik={acc:.4f}  TPR@FPR{a.target_fpr:.0%}={thr['tpr']:.4f}")

    results = {
        "task": a.task,
        "task_name": TASKS[a.task]["name"],
        "manifest": a.manifest,
        "cache": a.cache,
        "seed": a.seed,
        "train_profiles": a.profiles or "all",
        "selection": meta,
        "test_per_profile": per_profile,
        "WARNING": (
            "Gorev A icin: cozunurluk/metadata kestirme yolu ACIK "
            "(W3 Bulgu 9-10). Buradaki AUC forensic sinyal olarak "
            "yorumlanamaz." if a.task in ("A", "AB") else ""
        ),
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    probe.save(out_dir / "probe.joblib")
    print(f"\nKaydedildi: {out_dir}/results.json, probe.joblib")
    if results["WARNING"]:
        print(f"\n!!! {results['WARNING']}")


if __name__ == "__main__":
    _cli()
