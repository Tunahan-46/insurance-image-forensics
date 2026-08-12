"""
Rapor motoru: bir Detector + bir manifest split'i alır, standart bir
sonuç JSON'u ve grafikler üretir. Her deney (E0, E1, E2, ...) bu fonksiyonu
çağırır — böylece 17 deneyin hepsi aynı, karşılaştırılabilir formatta çıkar.

Kullanım:
    from src.detectors.base import BaseDetector, DetectorOutput
    from src.eval.report import run_and_report

    class MyDetector(BaseDetector):
        name = "my_detector_v1"
        def predict(self, image_path):
            ...

    run_and_report(
        detector=MyDetector(),
        manifest_path="data/processed/manifest.parquet",
        split="test",
        experiment_dir="experiments/E01_my_detector",
    )
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sunucu/sandbox ortamında ekran gerektirmez
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

from src.data.manifest import load_manifest
from src.detectors.base import Detector
from src.eval.metrics import compute_image_level_metrics


def _label_to_binary(label: str) -> int:
    """Manifest label'ını ikili (0=gerçek, 1=şüpheli) etikete çevirir."""
    return 0 if label == "real" else 1


def run_and_report(
    detector: Detector,
    manifest_path: str | Path,
    experiment_dir: str | Path,
    split: str = "test",
    launder_profiles: list[str] | None = None,
    max_samples: int | None = None,
) -> dict:
    """Ana giriş noktası. Her laundering profili için AYRI metrik hesaplar
    (bkz. plan 4.4 — 'sonuç tablon senaryo × profil matrisidir') ve hepsini
    tek bir JSON'da toplar.

    Döndürür: {"clean": {...metrikler...}, "whatsapp": {...}, ...}
    """
    experiment_dir = Path(experiment_dir)
    (experiment_dir / "plots").mkdir(parents=True, exist_ok=True)

    df = load_manifest(manifest_path)
    df = df[df["split"] == split].copy()
    if max_samples:
        df = df.sample(min(max_samples, len(df)), random_state=42)

    if launder_profiles is None:
        launder_profiles = sorted(df["launder_profile"].unique())

    all_results: dict[str, dict] = {}
    raw_scores: dict[str, dict] = {}

    for profile in launder_profiles:
        sub = df[df["launder_profile"] == profile]
        if len(sub) == 0:
            print(f"[uyarı] '{profile}' profili için örnek yok, atlanıyor.")
            continue

        print(f"[{detector.name}] '{profile}' profili değerlendiriliyor "
              f"({len(sub)} görüntü)...")

        outputs = detector.predict_batch(sub["path"].tolist())
        y_true = np.array([_label_to_binary(l) for l in sub["label"]])
        y_score = np.array([o.score for o in outputs])

        try:
            metrics = compute_image_level_metrics(y_true, y_score)
        except ValueError as e:
            print(f"[uyarı] '{profile}' için metrik hesaplanamadı: {e}")
            continue

        all_results[profile] = metrics.to_dict()
        raw_scores[profile] = {"y_true": y_true.tolist(), "y_score": y_score.tolist()}

        _plot_roc(y_true, y_score, profile, experiment_dir / "plots" / f"roc_{profile}.png")

    # Sonuç JSON'unu kaydet
    result_payload = {
        "detector_name": detector.name,
        "manifest": str(manifest_path),
        "split": split,
        "results_by_launder_profile": all_results,
    }
    with open(experiment_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    # Ham skorları da kaydet (sonradan başka analiz/ensemble için lazım olur)
    with open(experiment_dir / "raw_scores.json", "w", encoding="utf-8") as f:
        json.dump(raw_scores, f)

    _plot_summary_bar(all_results, detector.name, experiment_dir / "plots" / "summary_auc.png")

    print(f"\nRapor tamamlandı → {experiment_dir}/results.json")
    _print_summary_table(all_results)
    return result_payload


def _plot_roc(y_true: np.ndarray, y_score: np.ndarray, profile: str, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="rastgele")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {profile}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _plot_summary_bar(results: dict[str, dict], detector_name: str, out_path: Path) -> None:
    if not results:
        return
    profiles = list(results.keys())
    aucs = [results[p]["roc_auc"] for p in profiles]
    plt.figure(figsize=(6, 4))
    plt.bar(profiles, aucs, color="#4C72B0")
    plt.ylim(0, 1)
    plt.ylabel("ROC-AUC")
    plt.title(f"{detector_name} — profil bazında AUC")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _print_summary_table(results: dict[str, dict]) -> None:
    if not results:
        print("Hiç sonuç yok.")
        return
    print(f"\n{'profil':<15}{'AUC':>8}{'PR-AUC':>8}{'TPR@1%':>9}{'n':>7}")
    for profile, m in results.items():
        print(
            f"{profile:<15}{m['roc_auc']:>8.3f}{m['pr_auc']:>8.3f}"
            f"{m['tpr_at_fpr_1pct']:>9.3f}{m['n_samples']:>7}"
        )


if __name__ == "__main__":
    # Sanity check: sahte bir dedektör ve sahte bir manifest ile uçtan uca test.
    import tempfile
    from src.data.manifest import new_manifest, add_row, save_manifest
    from src.detectors.base import BaseDetector, DetectorOutput

    class RandomButBiasedDetector(BaseDetector):
        """Gerçek dedektörlere benzemesi için etiketle hafif korele rastgele skor."""

        name = "sanity_check_dummy"

        def __init__(self):
            self._rng = np.random.default_rng(0)

        def predict(self, image_path):
            # image_path'in içinde "fake" geçiyorsa yüksek skor eğilimi
            bias = 0.35 if "fake" in str(image_path) else 0.0
            score = float(np.clip(self._rng.normal(0.3 + bias, 0.2), 0, 1))
            return DetectorOutput(score=score)

    with tempfile.TemporaryDirectory() as tmpdir:
        df = new_manifest()
        for i in range(60):
            label = "real" if i % 2 == 0 else "fully_synthetic"
            path = f"data/fake_or_real_{i}.jpg"
            # Her laundering profilinde her iki sınıf da bulunmalı (i%4 ile
            # label ve profile'ı BAĞIMSIZ döndürüyoruz) — aksi halde
            # "y_true tek sınıf içeriyor" hatası alınır. Bu, gerçek
            # manifest'inde de kontrol etmen gereken bir tasarım kuralı.
            df = add_row(
                df,
                source_image_id=f"img_{i}",
                path=path,
                label=label,
                width=512,
                height=512,
                split="test",
                launder_profile="clean" if (i // 2) % 2 == 0 else "whatsapp",
            )
        manifest_path = Path(tmpdir) / "manifest.parquet"
        save_manifest(df, manifest_path)

        result = run_and_report(
            detector=RandomButBiasedDetector(),
            manifest_path=manifest_path,
            experiment_dir=Path(tmpdir) / "experiment_out",
            split="test",
        )
        assert "results_by_launder_profile" in result
        assert len(result["results_by_launder_profile"]) > 0

    print("\nreport.py sanity check OK")
