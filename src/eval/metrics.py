"""
Değerlendirme metrikleri.

Plan bölüm 8.1'deki tablo burada koda dökülüyor. Not:
  - Accuracy'yi tek başına RAPORLAMA (dengesiz veride yanıltıcı).
  - Ana operasyonel metrik: TPR @ FPR sabit (örn. %1). Bkz. plan 8.2.
  - Localization metrikleri (pixel F1 / IoU) SADECE gerçekten manipüle
    edilmiş görüntüler üzerinde hesaplanır — temiz görüntülerdeki
    "false positive area" ayrı bir fonksiyonla (fp_area_rate) ölçülür.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ImageLevelMetrics:
    """Görüntü-seviyesi (Task A ve Task B'nin detection kısmı) sonuç paketi."""

    n_samples: int
    n_positive: int
    roc_auc: float
    pr_auc: float
    tpr_at_fpr_1pct: float
    tpr_at_fpr_5pct: float
    ece: float  # Expected Calibration Error
    threshold_used: float
    precision_at_threshold: float
    recall_at_threshold: float
    f1_at_threshold: float
    confusion_matrix: list  # [[tn, fp], [fn, tp]]

    def to_dict(self) -> dict:
        return asdict(self)


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """Sabit bir yanlış-alarm bütçesinde (target_fpr) yakalanabilen TPR.
    Bu, plan 8.2'de anlatılan ana operasyonel metriktir:
    'günde X dosyadan Y'sini boşuna işaretlemeye razıyız, karşılığında
    sahtelerin ne kadarını yakalıyoruz?'"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(tpr[idx])


def expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> float:
    """Skorun gerçekten bir olasılık gibi davranıp davranmadığını ölçer.
    0.8 skoru gerçekten ~%80 olasılıkla doğru mu? Risk bantları (bkz.
    plan 7.1: DÜŞÜK/ORTA/YÜKSEK) ancak kalibre skorla anlamlıdır."""
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="uniform")
    # Basit, ağırlıksız ECE yaklaşımı (Hafta 5'te isotonic/Platt ile iyileşecek)
    return float(np.mean(np.abs(prob_true - prob_pred)))


def compute_image_level_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | None = None,
) -> ImageLevelMetrics:
    """Ana giriş noktası. y_true: 0/1 (1=şüpheli/sahte). y_score: [0,1] skor."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "y_true tek sınıf içeriyor — ROC/PR-AUC hesaplanamaz. "
            "Manifest filtrenizi kontrol edin (real+fake karışık mı?)."
        )

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    t1 = tpr_at_fpr(y_true, y_score, 0.01)
    t5 = tpr_at_fpr(y_true, y_score, 0.05)
    ece = expected_calibration_error(y_true, y_score)

    if threshold is None:
        # Varsayılan: Youden's J istatistiği ile ROC eğrisi üzerinde en iyi nokta.
        # NOT: Üretimde eşik iş kısıtına göre (örn. sabit FPR) seçilmeli,
        # bu sadece raporlama için bir varsayılan.
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j = tpr - fpr
        threshold = float(thresholds[np.argmax(j)])

    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return ImageLevelMetrics(
        n_samples=len(y_true),
        n_positive=int(y_true.sum()),
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
        tpr_at_fpr_1pct=t1,
        tpr_at_fpr_5pct=t5,
        ece=ece,
        threshold_used=float(threshold),
        precision_at_threshold=float(precision_score(y_true, y_pred, zero_division=0)),
        recall_at_threshold=float(recall_score(y_true, y_pred, zero_division=0)),
        f1_at_threshold=float(f1_score(y_true, y_pred, zero_division=0)),
        confusion_matrix=cm.tolist(),
    )


# ---------------------------------------------------------------------------
# Localization metrikleri (Hafta 4'te kullanılacak, şimdiden hazır)
# ---------------------------------------------------------------------------

def pixel_f1_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> dict:
    """SADECE gerçekten manipüle edilmiş görüntülerde çağır.
    pred_mask, gt_mask: aynı boyutta, [0,1] aralığında."""
    pred_bin = (pred_mask >= threshold).astype(np.uint8)
    gt_bin = (gt_mask >= threshold).astype(np.uint8)

    tp = int(np.sum((pred_bin == 1) & (gt_bin == 1)))
    fp = int(np.sum((pred_bin == 1) & (gt_bin == 0)))
    fn = int(np.sum((pred_bin == 0) & (gt_bin == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0

    return {"pixel_precision": precision, "pixel_recall": recall, "pixel_f1": f1, "iou": iou}


def fp_area_rate(pred_mask: np.ndarray, threshold: float = 0.5) -> float:
    """SADECE gerçek (temiz, manipüle edilmemiş) görüntülerde çağır.
    Modelin, hiçbir şey manipüle edilmemişken görüntünün ne kadarını
    'şüpheli' işaretlediğini ölçer. Yüksekse: gürültülü ısı haritası,
    eksper güveni kaybı (bkz. plan 8.1, 'FP area rate')."""
    pred_bin = (pred_mask >= threshold).astype(np.uint8)
    return float(np.mean(pred_bin))


if __name__ == "__main__":
    # Sanity check — sentetik ama gerçekçi bir dağılımla.
    rng = np.random.default_rng(42)
    n = 500
    y_true = rng.integers(0, 2, size=n)
    # Skor, gerçek etiketle ilişkili ama gürültülü — gerçekçi bir dedektör gibi
    y_score = np.clip(y_true * 0.5 + rng.normal(0.3, 0.25, size=n), 0, 1)

    m = compute_image_level_metrics(y_true, y_score)
    print("Image-level metrikler:")
    for k, v in m.to_dict().items():
        print(f"  {k}: {v}")

    # Localization sanity check
    gt = np.zeros((64, 64))
    gt[20:40, 20:40] = 1.0  # gerçek manipüle bölge
    pred = np.zeros((64, 64))
    pred[22:38, 18:42] = 0.9  # model biraz kaymış ama yakın tahmin

    loc = pixel_f1_iou(pred, gt)
    print("\nLocalization metrikleri (kaymış ama makul tahmin):")
    for k, v in loc.items():
        print(f"  {k}: {v:.3f}")

    clean_pred = np.zeros((64, 64))
    clean_pred[0:5, 0:5] = 0.6  # temiz görüntüde küçük bir FP alan
    print(f"\nFP area rate (temiz görüntü): {fp_area_rate(clean_pred):.4f}")

    assert 0.0 <= m.roc_auc <= 1.0
    assert 0.0 <= loc["pixel_f1"] <= 1.0
    print("\nmetrics.py sanity check OK")
