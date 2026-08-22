"""Olasilik kalibrasyonu (plan Hafta 3).

NEDEN KALIBRASYON
-----------------
Sistem karar verici degil TRIAGE (plan bolum 2): supheli dosyayi insan
eksperine yonlendiriyor. Boyle bir sistemde skorun SIRALAMASI kadar
BUYUKLUGU de anlamli olmali -- "%90 supheli" diyen bir cikti gercekten
10 dosyadan 9'unda hakli cikmali, yoksa eksper skora guvenmeyi birakir.

ROC-AUC bu konuda hicbir sey soylemez: AUC siralamaya bakar, mutlak
degerlere degil. Monoton bir donusum AUC'u degistirmez ama kalibrasyonu
tamamen bozabilir. Bu yuzden AUC'un yaninda ECE / reliability diyagrami
raporlanir.

IKI YONTEM
----------
  platt     : sigmoid(a*s + b), tek parametreli aile. Az veriyle saglam,
              az esnek. Kucuk val setlerinde varsayilan.
  isotonic  : parcali sabit monoton fonksiyon. Daha esnek ama VERI ISTER;
              kucuk val setinde asiri uyum yapar ve merdiven basamaklari
              uretir.

Bizim val setimiz profil basina 995 satir. Platt varsayilan; isotonic
karsilastirma icin var.

SIZINTI UYARISI
---------------
Kalibrasyon val uzerinde fit edilir, test uzerinde ASLA. Test'e bakip
kalibrasyon secmek plan 4.5 Tuzak 5'in ta kendisidir.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Kalibratorler
# ---------------------------------------------------------------------------

@dataclass
class PlattCalibrator:
    """sigmoid(a*s + b) -- Platt (1999) olceklemesi.

    Lojistik regresyonun tek ozellikli hali olarak kurulur; boylece
    sklearn'in saglam optimizasyonunu kullaniriz."""

    a: float = 1.0
    b: float = 0.0

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        s = np.asarray(scores, dtype=float).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(s, np.asarray(y).astype(int))
        self.a = float(lr.coef_[0][0])
        self.b = float(lr.intercept_[0])
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        return 1.0 / (1.0 + np.exp(-(self.a * s + self.b)))

    def to_dict(self) -> dict:
        return {"kind": "platt", "a": self.a, "b": self.b}

    @staticmethod
    def from_dict(d: dict) -> "PlattCalibrator":
        return PlattCalibrator(a=float(d["a"]), b=float(d["b"]))


class IsotonicCalibrator:
    """Parcali sabit monoton kalibrasyon. Karsilastirma amacli."""

    def __init__(self) -> None:
        self._iso = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(np.asarray(scores, dtype=float), np.asarray(y).astype(float))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("Once fit() cagir.")
        return np.asarray(self._iso.predict(np.asarray(scores, dtype=float)))


def fit_calibrator(scores, y, kind: str = "platt"):
    if kind == "platt":
        return PlattCalibrator().fit(scores, y)
    if kind == "isotonic":
        return IsotonicCalibrator().fit(scores, y)
    raise ValueError(f"Bilinmeyen kalibrasyon: {kind}")


# ---------------------------------------------------------------------------
# Olcum
# ---------------------------------------------------------------------------

def reliability_curve(
    probs: np.ndarray, y: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> dict:
    """Reliability diyagrami icin bin bazinda (guven, gerceklesme, n).

    strategy='quantile': her bin'de esit sayida ornek. Esit genislikli
    bin'lerde skorlar 0/1 uclarina yigildiginda ortadaki bin'ler bosalir
    ve diyagram yaniltici olur -- dengesiz sinifli forensic skorlarda tipik.
    """
    p = np.asarray(probs, dtype=float)
    t = np.asarray(y).astype(int)

    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])

    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    conf, acc, cnt = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        conf.append(float(p[m].mean()))
        acc.append(float(t[m].mean()))
        cnt.append(int(m.sum()))
    return {"confidence": conf, "empirical": acc, "count": cnt, "edges": edges.tolist()}


def plot_reliability(curves: dict[str, dict], out_path, title: str = "Kalibrasyon") -> None:
    """Bir veya birden fazla reliability egrisini tek grafige cizer.

    curves: {"kalibrasyonsuz": curve_dict, "platt": curve_dict, ...}
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="mukemmel kalibrasyon")
    for name, c in curves.items():
        ax.plot(c["confidence"], c["empirical"], "o-", ms=4, label=name)
    ax.set_xlabel("ortalama tahmin edilen olasilik")
    ax.set_ylabel("gozlenen sahte orani")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


__all__ = [
    "PlattCalibrator",
    "IsotonicCalibrator",
    "fit_calibrator",
    "reliability_curve",
    "plot_reliability",
]
