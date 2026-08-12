"""
L1 — Metadata / EXIF dedektörü.

Plan bölüm 3.2 ve 7.1'de anlatılan katman. Sıfır GPU, milisaniyeler,
tamamen açıklanabilir. Tek başına yeterli değil ama ucuz ve ilk filtre
olarak değerli.

KRİTİK KURAL (plan 7.1): Metadata HER ZAMAN orijinal yüklenen dosyadan
okunmalı, resize/format-dönüşümünden ÖNCE. Bu dosya bunu garanti eder:
predict() dosya yolunu doğrudan alır, hiçbir ön-işleme yapmaz.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.detectors.base import BaseDetector, DetectorOutput

# Bilinen düzenleme/AI araçlarının EXIF Software alanında bıraktığı izler.
# NOT: Bu liste kapsamlı değildir ve genişleyecektir — bir sinyaldir, kesin kanıt değil.
SUSPICIOUS_SOFTWARE_TAGS = [
    "photoshop",
    "gimp",
    "lightroom",
    "snapseed",
    "picsart",
    "facetune",
    "midjourney",
    "stable diffusion",
    "dall-e",
    "dalle",
]

# Tipik telefon kameralarının kuantizasyon tablosu davranışı vs. genel
# amaçlı kütüphanelerin (PIL/OpenCV varsayılanı) davranışı farklıdır.
# Burada tam bir kütüphane karşılaştırması yapmıyoruz (Hafta 5'te
# dq_flag/fft_score gibi feature'larla derinleştirilecek); şimdilik
# EXIF/tablo YOKLUĞUNU bir sinyal olarak kullanıyoruz.


def _read_exif(path: Path) -> dict:
    try:
        img = Image.open(path)
        exif = img.getexif()
        if not exif:
            return {}
        # EXIF tag ID'lerini insan-okunur isimlere çevir (temel bir alt küme)
        from PIL.ExifTags import TAGS

        return {TAGS.get(tag_id, tag_id): value for tag_id, value in exif.items()}
    except Exception:
        return {}


def _get_jpeg_quantization_tables(path: Path) -> list | None:
    """PIL, JPEG kuantizasyon tablosunu img.quantization üzerinden verir.
    Farklı yazılımlar (iPhone kamerası, Photoshop, PIL/Python) farklı
    tablolar üretir. Şimdilik ham tabloyu meta'ya koyuyoruz; Hafta 5'te
    bilinen imzalarla eşleştirme (dq_flag) eklenecek."""
    try:
        img = Image.open(path)
        if img.format != "JPEG":
            return None
        return getattr(img, "quantization", None)
    except Exception:
        return None


def analyze_metadata(path: str | Path) -> DetectorOutput:
    path = Path(path)
    exif = _read_exif(path)
    quant_tables = _get_jpeg_quantization_tables(path)

    flags: dict[str, bool | str] = {}
    score = 0.0  # 0 = şüphe yok, 1 = yüksek şüphe

    exif_present = len(exif) > 0
    flags["exif_present"] = exif_present
    if not exif_present:
        # EXIF yokluğu KESİN kanıt değildir (WhatsApp da siler — bkz. plan 3.2)
        # ama tek başına kural-tabanlı bir sinyal olarak hafif ağırlık taşır.
        score += 0.25
        flags["reason_no_exif"] = (
            "EXIF verisi yok — yeniden kaydedilmiş/düzenlenmiş olabilir "
            "(veya sadece WhatsApp gibi bir uygulamadan geçmiş olabilir)"
        )

    software = str(exif.get("Software", "")).lower()
    flags["software_tag"] = software or None
    if software:
        for tag in SUSPICIOUS_SOFTWARE_TAGS:
            if tag in software:
                score += 0.5
                flags["reason_suspicious_software"] = (
                    f"EXIF Software alanında şüpheli araç izi: '{software}'"
                )
                break

    make = exif.get("Make")
    model = exif.get("Model")
    flags["camera_make"] = make
    flags["camera_model"] = model
    has_camera_info = bool(make or model)
    flags["has_camera_info"] = has_camera_info
    if exif_present and not has_camera_info:
        # EXIF var ama kamera bilgisi yok → muhtemelen bir düzenleme
        # yazılımı tarafından yeniden yazılmış EXIF
        score += 0.15
        flags["reason_no_camera_info"] = (
            "EXIF mevcut ama kamera Make/Model bilgisi eksik"
        )

    flags["has_jpeg_quant_table"] = quant_tables is not None

    score = min(score, 1.0)

    return DetectorOutput(
        score=score,
        features={"quant_tables_present": quant_tables is not None},
        meta=flags,
    )


class MetadataDetector(BaseDetector):
    """Detector arayüzüne uyan sarmalayıcı (wrapper).
    src/eval/report.py ile doğrudan kullanılabilir."""

    name = "metadata_rule_based_v1"

    def predict(self, image_path: str | Path) -> DetectorOutput:
        return analyze_metadata(image_path)


if __name__ == "__main__":
    # Sanity check: geçici olarak EXIF'siz bir JPEG oluştur ve analiz et.
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "no_exif_test.jpg"
        Image.new("RGB", (100, 100), color=(128, 64, 32)).save(test_path, "JPEG")

        detector = MetadataDetector()
        result = detector.predict(test_path)

        print(f"Dedektör: {detector.name}")
        print(f"Skor: {result.score:.3f}")
        print("Bayraklar:")
        for k, v in result.meta.items():
            print(f"  {k}: {v}")

        assert 0.0 <= result.score <= 1.0
        assert result.meta["exif_present"] is False
        print("\nmetadata.py sanity check OK")
