"""
E0 — Sanity baseline: ResNet-50 fine-tune.

Plan Hafta 1 / E0: "Amaç doğruluk değil; manifest → dataloader → train →
eval → rapor JSON zincirinin çalıştığını görmek."

Bu dosya iki şey içerir:
  1. ManifestImageDataset — manifest.parquet'i okuyan bir torch Dataset
  2. CNNBaselineDetector — Detector arayüzüne uyan, eğitilmiş bir modeli
     sarmalayan sınıf (Hafta 3'te E1/E2 için de bu iskelet kullanılacak,
     sadece backbone değişecek)

NOT (sandbox): torchvision.models.resnet50(pretrained=True) internet
üzerinden ImageNet ağırlığı indirir. Bu sandbox'ın ağ izin listesinde
download.pytorch.org yok, yani bu dosya BURADA pretrained=False ile test
edilir (sadece kod akışını doğrulamak için). Gerçek eğitimi Colab/Kaggle'da
pretrained=True ile çalıştır — internet erişimleri açık.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

from src.data.manifest import load_manifest
from src.detectors.base import BaseDetector, DetectorOutput

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class ManifestImageDataset(Dataset):
    """manifest.parquet + split filtresi → (tensor, label) çiftleri.

    label: 0 = real, 1 = real-olmayan (fully_synthetic VEYA manipulated).
    Task A/Task B ayrımı Hafta 3/4'te ayrı manifest filtreleriyle yapılacak;
    bu E0 sanity testi için ikisi birlikte "anomali" sınıfı sayılır.
    """

    def __init__(self, manifest_path: str | Path, split: str, transform=None):
        df = load_manifest(manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform or default_transform()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)
        label = 0 if row["label"] == "real" else 1
        return image, label


def build_model(pretrained: bool = True) -> nn.Module:
    """ResNet-50, son katman 2-sınıflı (real / anomali) olarak değiştirilmiş."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def train_one_epoch(model, dataloader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)


class CNNBaselineDetector(BaseDetector):
    """Eğitilmiş bir ResNet-50 checkpoint'ini Detector arayüzüne sarmalar."""

    name = "resnet50_e0_sanity"

    def __init__(self, checkpoint_path: str | Path | None = None, device: str = "cpu"):
        self.device = device
        self.model = build_model(pretrained=(checkpoint_path is None))
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(state)
        self.model.to(device).eval()
        self.transform = default_transform()

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> DetectorOutput:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        prob_anomaly = torch.softmax(logits, dim=1)[0, 1].item()
        return DetectorOutput(score=float(prob_anomaly))


if __name__ == "__main__":
    # Sanity check: pretrained=False (internet gerektirmez), sahte veriyle
    # tek epoch, sonra Detector arayüzü üzerinden predict.
    import tempfile
    import numpy as np
    from src.data.manifest import new_manifest, add_row, save_manifest
    from torch.utils.data import DataLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        img_dir = tmpdir / "imgs"
        img_dir.mkdir()

        df = new_manifest()
        rng = np.random.default_rng(0)
        for i in range(16):
            label = "real" if i % 2 == 0 else "fully_synthetic"
            # Gerçek/sentetik ayrımını simüle etmek için farklı renk dağılımı
            color = (
                tuple(rng.integers(150, 255, size=3))
                if label == "real"
                else tuple(rng.integers(0, 100, size=3))
            )
            path = img_dir / f"img_{i}.jpg"
            Image.new("RGB", (64, 64), color=tuple(int(c) for c in color)).save(path)
            df = add_row(
                df,
                source_image_id=f"img_{i}",
                path=str(path),
                label=label,
                width=64,
                height=64,
                split="train",
                launder_profile="clean",
            )
        manifest_path = tmpdir / "manifest.parquet"
        save_manifest(df, manifest_path)

        print("Dataset + DataLoader kuruluyor...")
        dataset = ManifestImageDataset(manifest_path, split="train")
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        print("Model kuruluyor (pretrained=False, sandbox — internet yok)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model(pretrained=False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        print("1 epoch eğitim (sadece zincir testi, kalite önemsiz)...")
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch loss: {loss:.4f}")

        checkpoint_path = tmpdir / "e0_checkpoint.pt"
        torch.save(model.state_dict(), checkpoint_path)

        print("Detector arayüzü üzerinden tahmin testi...")
        detector = CNNBaselineDetector(checkpoint_path=checkpoint_path, device=device)
        sample_path = img_dir / "img_0.jpg"
        out = detector.predict(sample_path)
        print(f"Örnek tahmin: score={out.score:.3f}")

        assert 0.0 <= out.score <= 1.0
        print("\ncnn_baseline.py sanity check OK (pretrained=False modunda)")
        print(
            "NOT: Gerçek eğitim için Colab'da build_model(pretrained=True) kullan "
            "— sandbox'ta ImageNet ağırlığı indirilemez (ağ izin listesi kısıtlı)."
        )
