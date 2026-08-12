# insurance-image-forensics

Sigorta hasar fotoğraflarında AI-üretimi ve manipülasyon tespiti — Türkiye Sigorta AI Ar-Ge stajı.
Kapsam, mimari ve 6 haftalık plan için: `docs/mentorluk_plani.md` (mentor dokümanı, buraya kopyala).

---

## 0. Kurulum

```bash
# Sanal ortam
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Bağımlılıklar
pip install -r requirements.txt

# Sağlık kontrolü — her modülün kendi sanity check'ini çalıştırır
python -m src.detectors.base
python -m src.data.manifest
python -m src.eval.metrics
python -m src.eval.report
python -m src.detectors.metadata
python -m src.detectors.cnn_baseline   # torch/torchvision gerektirir
```

Hepsi `... sanity check OK` ile bitiyorsa iskelet hazır demektir.

> **Not (GPU/Colab):** `cnn_baseline.py`'daki `build_model(pretrained=True)`
> ImageNet ağırlığını internetten indirir. Kısıtlı ağ ortamlarında
> (bazı kurumsal/sandbox ağlar) bu indirme engellenebilir — o durumda
> Google Colab veya Kaggle Notebooks kullan (ikisi de ücretsiz GPU verir
> ve tam internet erişimine sahiptir).

---

## 1. Hafta 1 — Neyi kendin indirmen/yapman gerekiyor

Aşağıdaki adımlar **senin kendi makinende veya Colab'da** yapılmalı (bu
kod iskeleti hangi ortamda hazırlandıysa oradan bazı sitelere ağ erişimi
kısıtlıydı — CarDD ve Hugging Face gibi siteler dahil).

### 1.1. CarDD veri setini indir

1. https://cardd-ustc.github.io adresine git, "Download" bağlantısını takip et
   (Google Drive veya Baidu NetDisk linki verilir).
2. İndirilen arşivi `data/raw/cardd/` altına çıkar. Beklenen yapı:
   ```
   data/raw/cardd/
   ├── CarDD_COCO/          # object detection / segmentation formatı
   ├── CarDD_SOD/           # salient object detection formatı
   └── ...
   ```
3. Lisans/kullanım şartlarını oku, `docs/licenses.md`'ye bir satır ekle
   (akademik/staj kullanımı genelde serbesttir ama kaydı tut).
4. Görüntü sayısını doğrula:
   ```bash
   find data/raw/cardd -iname "*.jpg" | wc -l   # ~4000 civarı beklenir
   ```

### 1.2. Kendi telefon fotoğraflarını çek (ÖNEMLİ — plan madde 4.2)

- Bir otoparkta 1 saat ayır, 50–100 araç fotoğrafı çek (çizik, göçük, sağlam
  paneller — karışık).
- Fotoğrafları **hiç düzenlemeden**, orijinal EXIF'iyle telefondan bilgisayara
  aktar (AirDrop/kablo kullan; WhatsApp/Telegram üzerinden GÖNDERME — EXIF siler).
- `data/raw/own_photos/` altına koy.
- Bu, projendeki **tek gerçek EXIF'li gerçek veri** olacak — metadata
  katmanının (L1) doğrulanması için altın standart.

### 1.3. Roboflow Universe araç hasar setleri (opsiyonel, hacim için)

- https://universe.roboflow.com adresinde "car damage" ara.
- Birkaç açık kaynak seti indir (COCO veya YOLO formatında), `data/raw/roboflow/`
  altına koy. Kaliteyi gözden geçir, bozuk/duplike olanları ele.

### 1.4. Hızlı sentetik görüntü üretimi (20 adet, Hafta 1 için yeterli)

Bu adım **Colab'da** çalıştırılmalı (diffusers modeli Hugging Face'ten
indirir, sandbox bunu yapamaz). `notebooks/week1_quick_synth.ipynb`
dosyasını oluştur (veya aşağıdaki hücreleri Colab'a yapıştır):

```python
# Colab hücresi 1 — kurulum
!pip install diffusers accelerate safetensors -q

# Colab hücresi 2 — hızlı üretim (SD 1.5, düşük adım sayısı = hızlı test)
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

prompts = [
    "insurance claim photo, silver hatchback, deep scratch on the driver "
    "side door, apartment parking garage, fluorescent lighting, close-up "
    "45 degree angle, shot on Android phone, slight motion blur",
    # ... plan bölüm 4.3'teki şablon motorunu kullanarak 20 farklı prompt üret
]
negative = ("professional, cinematic, 8k, artstation, illustration, render, "
            "cartoon, oversaturated, studio lighting, perfect composition")

for i, p in enumerate(prompts):
    image = pipe(p, negative_prompt=negative, num_inference_steps=25).images[0]
    image.save(f"synth_{i:03d}.jpg")
```

Ürettiğin 20 görüntüyü indirip `data/raw/synthetic_quick/` altına koy.
**Bunları gözünle kontrol et** — bariz bozuk olanları eleme (plan 4.3).

### 1.5. Manifest'i doldur

`data/raw/` altındaki gerçek klasör yapısını gördükten sonra, bu klasörleri
tarayıp `src/data/manifest.py`'deki `add_row()` ile bir manifest üreten
küçük bir script yaz: `scripts/build_manifest_v1.py`. Şablon:

```python
from pathlib import Path
from PIL import Image
from src.data.manifest import new_manifest, add_row, save_manifest, check_split_leakage

df = new_manifest()

# --- CarDD (real) ---
cardd_images = list(Path("data/raw/cardd").rglob("*.jpg"))
for i, p in enumerate(cardd_images):
    w, h = Image.open(p).size
    split = "test" if i % 10 == 0 else ("val" if i % 10 == 1 else "train")
    df = add_row(df, source_image_id=f"cardd_{i:05d}", path=str(p),
                 label="real", width=w, height=h, split=split,
                 launder_profile="clean")

# --- Kendi fotoğrafların (real) ---
own_images = list(Path("data/raw/own_photos").glob("*.jpg"))
for i, p in enumerate(own_images):
    w, h = Image.open(p).size
    df = add_row(df, source_image_id=f"own_{i:03d}", path=str(p),
                 label="real", width=w, height=h, split="test",  # test'e ağırlık ver
                 launder_profile="clean")

# --- Hızlı sentetik (fully_synthetic) ---
synth_images = list(Path("data/raw/synthetic_quick").glob("*.jpg"))
for i, p in enumerate(synth_images):
    w, h = Image.open(p).size
    split = "test" if i % 5 == 0 else "train"
    df = add_row(df, source_image_id=f"synth_{i:03d}", path=str(p),
                 label="fully_synthetic", generator="sd15", width=w, height=h,
                 split=split, launder_profile="clean")

problems = check_split_leakage(df)
if problems:
    raise RuntimeError(problems)

save_manifest(df, "data/processed/manifest_v1.parquet")
print(f"Toplam {len(df)} satır, sızıntı kontrolü temiz.")
```

### 1.6. E0'ı gerçek veriyle çalıştır

```bash
python -m src.eval.report   # önce sanity check ile mekaniği doğrula
```

Sonra kendi script'ini yaz (`scripts/run_e0.py`):
```python
from src.detectors.cnn_baseline import CNNBaselineDetector
from src.eval.report import run_and_report

# Not: gerçek eğitim Colab'da yapılıp checkpoint indirilecek,
# ya da pretrained=True ile sıfır-atış (zero-shot ImageNet) test edilecek.
detector = CNNBaselineDetector(checkpoint_path=None, device="cpu")
run_and_report(
    detector=detector,
    manifest_path="data/processed/manifest_v1.parquet",
    experiment_dir="experiments/E00_sanity",
    split="test",
)
```

---

## 2. Proje yapısı

```
insurance-image-forensics/
├── src/
│   ├── data/manifest.py          # ✅ Hafta 1 — veri seti kayıt sistemi
│   ├── detectors/
│   │   ├── base.py               # ✅ Hafta 1 — ortak Detector arayüzü
│   │   ├── metadata.py           # ✅ Hafta 1 — L1 EXIF/metadata katmanı
│   │   └── cnn_baseline.py       # ✅ Hafta 1 — E0 sanity ResNet-50
│   ├── eval/
│   │   ├── metrics.py            # ✅ Hafta 1 — ROC-AUC, TPR@FPR, ECE, IoU
│   │   └── report.py             # ✅ Hafta 1 — otomatik deney raporlama
│   ├── features/, fusion/, explain/, pipeline/, api/   # Hafta 3-6
├── data/raw/                     # sen dolduracaksın (bkz. yukarı)
├── data/processed/                # manifest'ler ve laundering çıktıları
├── experiments/                  # her deney kendi klasöründe
├── docs/weekly/                  # haftalık 1 sayfa raporlar
├── scripts/                      # build_manifest_v1.py vb. senin yazacakların
└── requirements.txt
```

## 3. Hafta 1 bitiş kriterleri (checklist)

- [x] Repo iskeleti ve `Detector` arayüzü
- [x] Manifest sistemi + sızıntı kontrolü (`check_split_leakage`)
- [x] Metrik altyapısı (image-level + localization)
- [x] Otomatik rapor motoru (`run_and_report`)
- [x] L1 metadata dedektörü
- [x] E0 sanity CNN baseline (pipeline testi)
- [ ] **Sen:** CarDD indir, kendi fotoğraflarını çek, 20 sentetik üret (Colab)
- [ ] **Sen:** `scripts/build_manifest_v1.py` yaz, gerçek manifest'i üret
- [ ] **Sen:** `check_split_leakage` ve `check_generator_disjoint` ile doğrula
- [ ] **Sen:** E0'ı gerçek veriyle çalıştır, `experiments/E00_sanity/results.json` üret
- [ ] **Sen:** `docs/weekly/W1.md` yaz (ne yaptım / ne öğrendim / gelecek hafta)
- [ ] **Sen:** 6 makaleyi oku, `docs/lit/` altına notları düş
