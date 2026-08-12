"""
Ortak Detector arayüzü.

Bu proje boyunca yazılacak HER dedektör (metadata, CLIP synthetic detection,
TruFor localization, klasik forensics...) bu sözleşmeye uyar. Böylece:

  - src/eval/report.py tek bir kod yoluyla her dedektörü ölçebilir
  - src/fusion/feature_builder.py katmanları birbirinin yerine takıp çıkarabilir
  - Yeni bir model eklemek "yeni bir dosya + bu arayüzü uygula" kadar basit olur

Tasarım kararı: DetectorOutput hem görüntü-seviyesi skor (score) hem de
opsiyonel piksel-seviyesi maske (mask) taşır. Task A (synthetic detection)
dedektörleri mask=None döner; Task B (localization) dedektörleri hem score
hem mask döner (score genelde mask'tan türetilir, bkz. E11).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np


@dataclass
class DetectorOutput:
    """Bir dedektörün tek bir görüntü için ürettiği standart çıktı."""

    # 0-1 arası "şüpheli/sahte/manipüle" olasılığı. Yüksek = daha şüpheli.
    score: float

    # Piksel-seviyesi manipülasyon maskesi (H, W), değerler 0-1.
    # Synthetic-detection dedektörleri (CLIP probe, CNN baseline) için None.
    mask: Optional[np.ndarray] = None

    # Fusion katmanına (Hafta 5) girecek ham/ara özellikler.
    # Örn: {"clip_embedding": np.ndarray} veya {"jpeg_quality_est": 78}
    features: dict[str, Any] = field(default_factory=dict)

    # Hata ayıklama / açıklanabilirlik için serbest metin ve bayraklar.
    # Örn: {"exif_present": False, "software_tag": "Adobe Photoshop 26.0"}
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score [0,1] aralığında olmalı, geldi: {self.score}")
        if self.mask is not None and self.mask.ndim != 2:
            raise ValueError(f"mask 2 boyutlu (H,W) olmalı, geldi: {self.mask.shape}")


class Detector(Protocol):
    """Her dedektörün uyması gereken minimal arayüz."""

    name: str  # Rapor/loglarda görünecek kısa kimlik, örn. "clip_vitl_linear_probe"

    def predict(self, image_path: str | Path) -> DetectorOutput:
        """Tek bir görüntü için DetectorOutput üretir."""
        ...

    def predict_batch(self, image_paths: list[str | Path]) -> list[DetectorOutput]:
        """Varsayılan implementasyon: predict()'i sırayla çağırır.
        GPU'lu dedektörler (CLIP, TruFor) bunu gerçek batch'leme ile
        override ederek hız kazanabilir — zorunlu değil, opsiyonel."""
        ...


class BaseDetector:
    """Protocol'ü uygulayan, predict_batch'i bedava veren temel sınıf.
    Yeni bir dedektör yazarken bundan miras al, sadece predict()'i doldur."""

    name: str = "base_detector"

    def predict(self, image_path: str | Path) -> DetectorOutput:
        raise NotImplementedError

    def predict_batch(self, image_paths: list[str | Path]) -> list[DetectorOutput]:
        return [self.predict(p) for p in image_paths]

    def __repr__(self) -> str:
        return f"<Detector: {self.name}>"


if __name__ == "__main__":
    # Hızlı sanity check — arayüzün doğru çalıştığını görmek için.
    class DummyDetector(BaseDetector):
        name = "dummy_always_0.5"

        def predict(self, image_path):
            return DetectorOutput(score=0.5, meta={"path": str(image_path)})

    d = DummyDetector()
    out = d.predict("fake/path.jpg")
    print(d)
    print(out)
    batch = d.predict_batch(["a.jpg", "b.jpg"])
    print(f"Batch sonucu: {len(batch)} adet DetectorOutput")
    print("base.py sanity check OK")
