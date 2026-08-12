"""
E0 - Sanity baseline deneyi.

Amac (plan Hafta 1): model kalitesi DEGIL, zincirin calistigini gormek.
    manifest -> detector -> metrik -> results.json + grafikler

Iki dedektor calistirilir:
  1. MetadataDetector (L1)  - GPU gerektirmez, hemen calisir
  2. CNNBaselineDetector    - ImageNet pretrained ResNet-50, egitilmemis
                              (zero-shot). Skorlari anlamsiz olacak, bu NORMAL.
                              Amac sadece torch/dataloader zincirini test etmek.

Calistirma:
    python scripts/run_e0.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detectors.metadata import MetadataDetector  # noqa: E402
from src.eval.report import run_and_report  # noqa: E402

MANIFEST = "data/processed/manifest_v1.parquet"


def run_metadata_detector() -> None:
    print("=" * 60)
    print("E0-a: Metadata dedektoru (L1)")
    print("=" * 60)
    run_and_report(
        detector=MetadataDetector(),
        manifest_path=MANIFEST,
        experiment_dir="experiments/E00_metadata",
        split="test",
    )
    print("\nYORUM: Bu dedektor sadece EXIF/JPEG bilgisine bakiyor.")
    print("CarDD goruntuleri yeniden kaydedilmis oldugu icin EXIF'siz olabilir;")
    print("kendi telefon fotograflarin ise tam EXIF'li. Skorlarin bu ayrimi")
    print("yansitmasi BEKLENEN davranistir - henuz 'sahte tespiti' degil.")


def run_cnn_baseline() -> None:
    print("\n" + "=" * 60)
    print("E0-b: CNN baseline (ResNet-50, egitilmemis)")
    print("=" * 60)
    try:
        from src.detectors.cnn_baseline import CNNBaselineDetector
    except ImportError as e:
        print(f"torch yuklenemedi, bu adim atlaniyor: {e}")
        return

    print("ImageNet agirliklari indiriliyor (ilk calistirmada ~100MB)...")
    try:
        detector = CNNBaselineDetector(checkpoint_path=None, device="cpu")
    except Exception as e:
        print(f"Model yuklenemedi: {e}")
        print("Internet baglantisini kontrol et veya bu adimi Colab'da calistir.")
        return

    # NOT: max_samples KULLANILMIYOR. Rastgele ornekleme, azinlik sinifin
    # (sentetik) tamamen disarida kalmasina ve "tek sinif" hatasina yol
    # acabiliyor. Tum test setini calistiriyoruz - CPU'da birkac dakika
    # surer ama sonuc guvenilir olur.
    run_and_report(
        detector=detector,
        manifest_path=MANIFEST,
        experiment_dir="experiments/E00_cnn_sanity",
        split="test",
    )
    print("\nYORUM: Bu model real/fake ayrimi icin HIC EGITILMEDI.")
    print("AUC'un 0.5 civarinda cikmasi BEKLENEN ve DOGRU sonuctur.")
    print("Buradaki amac sadece torch -> dataloader -> metrik zincirini")
    print("dogrulamak. Gercek egitim Hafta 3'te (E1/E3) yapilacak.")


def main() -> None:
    if not Path(MANIFEST).exists():
        print(f"HATA: {MANIFEST} bulunamadi.")
        print("Once calistir: python scripts/build_manifest_v1.py")
        sys.exit(1)

    run_metadata_detector()
    run_cnn_baseline()

    print("\n" + "=" * 60)
    print("E0 TAMAMLANDI")
    print("=" * 60)
    print("Uretilen dosyalar:")
    for exp in ["E00_metadata", "E00_cnn_sanity"]:
        d = Path("experiments") / exp
        if d.exists():
            print(f"  {d}/results.json")
            for p in sorted((d / "plots").glob("*.png")):
                print(f"  {p}")
    print("\nSonraki adim: docs/weekly/W1.md yaz, sonra commit + push.")


if __name__ == "__main__":
    main()