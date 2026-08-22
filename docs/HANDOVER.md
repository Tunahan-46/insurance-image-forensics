# DEVİR TESLİM — insurance-image-forensics

> Bu belge, sohbet bağlamı sıfırlandığında projeyi devralacak bir asistanın
> hiçbir şey sormadan çalışmaya başlayabilmesi için yazıldı. Rakamlar tahmin
> değil; `manifest_v2.parquet`, `manifest_v2_laundered.parquet` ve
> `experiments/` çıktılarından ölçüldü.
>
> **Son güncelleme:** Hafta 3 sonu. Kritik durum: veri sızıntısı düzeltmesi
> **kod olarak yazıldı ama HENÜZ ÇALIŞTIRILMADI.** Bkz. bölüm 5.

---

## 0. Ortam ve temel bilgiler

| | |
|---|---|
| Repo | `github.com/Tunahan-46/insurance-image-forensics` |
| Yerel yol | `C:\Users\tunah\OneDrive\Masaüstü\insurance-image-forensics` |
| İşletim sistemi | **Windows — komutlar PowerShell sözdiziminde verilmeli** |
| Plan belgesi | `docs/mentorluk_plani.md` (1158 satır, 6 haftalık yol haritası — **tek doğruluk kaynağı**) |
| Durum analizi | `docs/durum_analizi_H3.md` (bu oturumda üretildi, plan vs gerçek denetimi) |
| GPU işi | Kaggle **T4 × 2** notebook'ları (yerel GPU yok) |
| Dil | Kullanıcı Türkçe konuşuyor; kod yorumları da Türkçe (ASCII, Türkçe karakter yok) |

**Bilinen ortam tuzağı:** OpenCV 5.0 `cv2.CascadeClassifier`'ı kaldırdı →
plaka bulanıklaştırma otomatik yapılamıyor. KVKK gereği kendi çektiği 52
fotoğraf kolajlardan varsayılan olarak dışlanıyor (`--include-own` ile dahil
edilebilir).

---

## 1. Projenin mevcut durumu ve hedefi

### 1.1 Amaç

Sigorta hasar dosyalarına yüklenen **araç dış hasar fotoğraflarının** yapay
zekâ ile üretilmiş veya manipüle edilmiş olup olmadığını tespit etmek.
Kapsam dışı: konut/sağlık hasarı, fatura görselleri, video, yüz deepfake.

**İki ayrı görev:**

- **Task A** — görüntü seviyesinde *tam sentetik* tespiti (Hafta 2–3)
- **Task B** — *manipülasyon lokalizasyonu*, piksel maskesi üretimi (Hafta 4–5)
- Plan açıkça diyor: **"Sigorta açısından Task B daha kritiktir."**

### 1.2 Mimari (plan §1.4) — henüz tamamı kurulmadı

```
L1 METADATA          EXIF, cihaz/yazılım, JPEG kuantizasyon tablosu, C2PA  -> p_meta
L2 SYNTHETIC DET.    CLIP ViT-L/14 frozen + linear probe                   -> p_synthetic
L3 LOCALIZATION      TruFor / IML-ViT (zero-shot)                          -> p_manip + maske
L4 KLASİK FORENSICS  ELA / gürültü / FFT        (plan: "SINIRLI AL")
L5 FUSION            LogisticRegression + LightGBM, Platt/Isotonic
L6 DECISION          risk bandı + insan-okunur gerekçe + ısı haritası
```

**Üç tasarım ilkesi (pazarlık edilemez):**
1. **Karar değil triage.** Sistem "sahtedir" demez, risk bandı üretir.
   `<0.30` DÜŞÜK → otomatik akış · `0.30–0.70` ORTA → eksper · `≥0.70` YÜKSEK → derin inceleme.
2. **Kanıt üretir**, sadece skor değil.
3. **Modüler** — her katman bağımsız ölçülür.

### 1.3 Veri setinin ölçülmüş durumu

`data/processed/manifest_v2.parquet` — **5.222 satır** (clean katman):

| Etiket | Adet | Detay |
|---|---:|---|
| `real` | 4.052 | CarDD 4.000 + kendi telefon fotoğrafları 52 |
| `fully_synthetic` | 330 | sdxl 80, sd15 150, sd_turbo 100 |
| `manipulated` | 840 | splice 225, copy_move 185, inpaint_remove 120, bg_replace 110, inpaint_add 108, inpaint_enlarge 92 |

Maskeli örnek: **840** · Split: train 3.558 / val 1.014 / test 650

`data/processed/manifest_v2_laundered.parquet` — **16.966 değerlendirme örneği**
(clean/whatsapp/double_jpeg tüm split'lerde; screenshot/aggressive yalnız test'te).

### 1.4 Plan hedefi vs gerçek — dürüst tablo

| Katman | Hedef | Gerçek | Oran |
|---|---:|---:|---:|
| R gerçek | 2.000 | 4.052 | %203 |
| S tam sentetik | 1.200 | **330** | **%28** |
| M1 inpaint ekleme/büyütme | 800 | **200** | **%25** |
| M2 inpaint silme | 400 | **120** | **%30** |
| M3 klasik | 400 | **520** | %130 |
| **Toplam sahte** | **2.800** | **1.170** | **%42** |

**FLUX hiç üretilmedi.** Planın E6 tarifi ("train SD1.5+SDXL, test FLUX")
harfiyen uygulanamaz. **Ancak `sd_turbo` zaten %100 test'e ayrılmış**
(train 0 / val 0 / test 100) — generator-disjoint protokol fiilen kurulu.
E6 bu ikame ile koşulmalı, raporda "FLUX yerine SD-Turbo" denmeli.

### 1.5 Haftalık ilerleme

| Hafta | Durum | Eksik olan |
|---|---|---|
| **Hafta 1** | **%95** | MLflow kurulumu |
| **Hafta 2** | **%65** | veri hacmi %42, test seti yeniden dondurulmalı, **geometri sızıntısı açık**, insan baseline'ı yok |
| **Hafta 3** | **%15** | **E1–E6'nın hiçbiri çalıştırılmadı**, gradcam.py yok |

> **En önemli cümle:** Üç haftanın sonunda **ölçülmüş tek bir model sonucu
> yok.** Planın "asla kesme" (P0) listesinin ilk maddesi olan CLIP baseline'ı
> (E3) kodu yazılı ve test edilmiş halde duruyor ama hiç koşulmadı.

---

## 2. Kullanılan teknolojiler, hiperparametreler, metrikler

### 2.1 Yığın

`torch`, `torchvision`, `timm`, `transformers`, `open_clip_torch` ·
`opencv-python-headless`, `Pillow`, `scikit-image`, `exifread`, `piexif`, `pillow-heif` ·
`scikit-learn`, `lightgbm`, `pandas`, `pyarrow`, `matplotlib` ·
`mlflow` (kurulu ama **hiç kullanılmıyor** — plan "ZORUNLU" diyor, sapma raporlanmalı) ·
`fastapi`, `uvicorn`, `gradio`, `pydantic` (Hafta 6) ·
`diffusers`, `accelerate`, `safetensors` (yalnız Kaggle'da)

### 2.2 Model yapılandırmaları

**E3 — ana baseline (`src/detectors/clip_probe.py`, yazıldı, koşulmadı):**
- Backbone: **CLIP ViT-L/14**, tüm ağırlıklar **dondurulmuş**
- Embedding: **768-d**, `.npy` shard cache (SHARD=2000), resume-safe
- Head: `sklearn.LogisticRegression(class_weight='balanced')`
- Grid: `C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`
- **Model seçimi: TPR@FPR=1%'e göre, eşitlikte AUC'ye göre** (AUC tek başına değil)
- Kalibrasyon: **Platt scaling** (val setinde fit)
- Görevler: `TASKS = {A: synthetic, B: manipulated, AB: any fake}`

**E0 (koşuldu, tasarımı gereği anlamsız):** ResNet-50 ImageNet-pretrained,
son katman fine-tune, 5 epoch → AUC 0.36. Amacı yalnızca pipeline'ın uçtan
uca döndüğünü kanıtlamak.

### 2.3 Laundering profilleri (`src/data/launder.py`)

| Profil | İşlem | JPEG q |
|---|---|---:|
| `clean` | (yalnız yeniden kaydetme) | 95 |
| `whatsapp` | uzun kenar 1600 | 75 |
| `screenshot` | 1280 + %2 kırpma + PNG roundtrip | 90 |
| `double_jpeg` | q95 roundtrip | 70 |
| `aggressive` | 1024 + gaussian blur 0.5 | 60 |

**YENİ (bu oturum):** her profilin **ilk adımı** artık geometri
normalizasyonu — merkezden kare kırpma + **448×448**. Bkz. bölüm 4.6.

`TRAIN_AUGMENT_PROFILES = (clean, whatsapp, double_jpeg)` — test'te 5 profilin hepsi.

### 2.4 Metrikler (`src/eval/metrics.py`)

- **TPR@FPR=1% — operasyonel ANA metrik** (plan §8.1). Accuracy neredeyse anlamsız, sunumda tek başına gösterilmeyecek.
- ROC-AUC (model karşılaştırma), PR-AUC (sahte oranı düşük olduğu için daha dürüst)
- ECE / kalibrasyon eğrisi · Piksel F1, IoU (Task B) · **FP area rate** (temiz görüntüde işaretlenen alan)

### 2.5 Taahhüt edilen hedefler (plan §8.3 — "dürüstlük sigortası")

| Hedef | Min | İyi | Mükemmel |
|---|---|---|---|
| Tam sentetik, `clean` | AUC 0.90 | 0.95 | 0.98 |
| Tam sentetik, `whatsapp` | 0.75 | 0.85 | 0.92 |
| Tam sentetik, **görülmemiş generator** | 0.70 | 0.82 | 0.90 |
| Lokal inpaint, `clean` | 0.70 | 0.82 | 0.90 |
| Localization piksel-F1 | 0.35 | 0.50 | 0.65 |
| Uçtan uca gecikme | <5 sn | <3 sn | <1.5 sn |

### 2.6 Deney disiplini (plan şartları — şu an İHLAL EDİLİYOR)

1. Her deneyde tek bir şey değiştir
2. **En az 3 seed, ortalama ± std** ← şu an tek seed
3. Her deney sonunda `NOTES.md`'ye "Hipotez: X. Sonuç: doğrulandı/reddedildi çünkü Y."
4. **Negatif sonuçları silme**
5. `experiments/EXX_isim/{config.yaml, run.py, results.json, plots/, NOTES.md}` ← şu an yalnız `results.json` var

---

## 3. Kritik kodlar — kaybedilmemesi gerekenler

### 3.1 Dosya envanteri

```
src/data/manifest.py          Parquet manifest + check_split_leakage()  [KRİTİK]
src/data/launder.py           5 profil + geometri normalizasyonu        [BU OTURUMDA DEĞİŞTİ]
src/data/masks.py             maske üretimi + yeni yardımcılar          [BU OTURUMDA DEĞİŞTİ]
src/data/imageio.py           imread/imwrite (Türkçe yol güvenli)
src/data/generators/
    fully_synthetic.py        SDXL/SD1.5/SD-Turbo
    inpaint_add.py            M1: hasar ekleme + büyütme
    inpaint_remove.py         M2: hasar/nesne silme
    classic_manip.py          M3: copy_move/splice/bg_replace           [BU OTURUMDA DEĞİŞTİ]
    prompts.py                kombinatoryal prompt motoru + negative prompt
src/detectors/base.py         Detector Protocol, DetectorOutput
src/detectors/metadata.py     L1 — EXIF/quant tablo/C2PA
src/detectors/cnn_baseline.py E0/E1/E2
src/detectors/clip_probe.py   E3 ANA BASELINE                          [YAZILDI, KOŞULMADI]
src/features/clip_embed.py    768-d shard cache                        [YAZILDI, KOŞULMADI]
src/eval/metrics.py           TPR@FPR, ECE, piksel F1/IoU
src/eval/calibration.py       Platt + Isotonic + reliability curve      [YAZILDI, KULLANILMADI]
src/eval/report.py            standart sonuç JSON + grafik

scripts/build_manifest_v2.py  manifest + --freeze-test
scripts/apply_laundering.py   laundering uygulayıcı                     [BU OTURUMDA DEĞİŞTİ]
scripts/check_leakage.py      geometri sızıntı kapısı                   [BU OTURUMDA YENİ]
scripts/run_e1_shortcut.py    meta/px8/px32 kestirme yol teşhisi
scripts/diag_splice.py        harmanlama kalite teşhisi (dE)            [BU OTURUMDA YENİ]
scripts/diag_bg_shape.py      bg_replace şekil teşhisi                  [BU OTURUMDA YENİ]
scripts/make_collage.py       senaryo kolajları (KVKK filtreli)
notebooks/W3_kaggle_clip_embed.ipynb   CLIP embedding çıkarımı          [YAZILDI, KOŞULMADI]
tests/test_week1_skeleton.py  13 pytest testi (hepsi geçiyor)
```

### 3.2 Sızıntı koruması — projenin omurgası

`src/data/manifest.py::check_split_leakage()` üç kuralı zorlar ve ihlalde
`apply_laundering.py` manifest'i **KAYDETMEDEN** çıkar:

1. **source-image-disjoint** — aynı kaynak fotoğrafın türevleri aynı split'te
2. **generator-disjoint** — sd_turbo yalnız test'te
3. **image_id benzersizliği**

**`classic_manip.py::generate()` içindeki `pick_donor()` — silinmemesi gereken
bir hata düzeltmesi.** splice/bg_replace iki kaynak kullanır (hedef + donör).
Donör rastgele seçilince ilk üretimde ölçülen sonuç: 116 grup çatıştı,
206 gerçek görüntü train/val'den test'e sürüklendi, test'te splice 91 /
bg_replace 41 birikti, val'de yok oldu. Çözüm: donör **baştan aynı split'ten**
seçiliyor. Bu mantık kaldırılırsa hata sessizce geri gelir.

### 3.3 Kalite kapıları (üretimde diske yazmayı engelleyenler)

```python
# src/data/generators/classic_manip.py
MIN_CHANGED_IN_MASK   = 0.25   # maske içi piksellerin >=%25'i gerçekten değişmeli
MAX_COLOR_DE          = 25.0   # splice/copy_move: maske içi <-> çevre renk mesafesi
BG_TONE_MATCH_STRENGTH= 0.5    # bg_replace: donör arka planın kısmi renk uyumu
BG_MAX_COLOR_ARTIS    = 30.0   # bg_replace: manip dE - kaynak dE (MUTLAK değil ARTIŞ)

# src/data/launder.py
NORMALIZE_EDGE        = 448    # merkezden kare kırpma + 448x448 (bkz. 4.6)
```

`bg_replace` neden **artış** ölçüyor da mutlak dE değil: bu senaryoda araç/arka
plan sınırında gerçek fotoğraflarda bile yüksek renk farkı OLMASI beklenir
(doğal taban dE ~16–40). Mutlak eşik koymak doğru örnekleri elerdi.

### 3.4 `masks.py` — bu oturumda eklenen üç fonksiyon

```python
shape_is_rectangular(mask, extent_thresh=0.92)
    # extent = kontur alanı / sınırlayıcı dikdörtgen alanı
    # GrabCut fallback'e düşmüş segmentasyonu yakalar

color_transfer(src, src_mask, ref, ref_mask, strength=0.85)
    # Reinhard 2001 Lab uzayında ortalama/std eşleme

color_consistency_de(img, mask, ring_inner=6, ring_outer=26)
    # maske içi <-> çevreleyen halka arasında CIE76 benzeri dE

vehicle_region(img, max_dim=512, hint_mask=None)   # ← hint_mask BU OTURUMDA EKLENDİ
    # hint_mask verilirse CarDD hasar konumu GrabCut'a GC_FGD tohumu olarak
    # geçilir (GC_INIT_WITH_MASK). Doğru nesneye kilitlenmeyi sağlar.
```

---

## 4. Çözülen hatalar — aynı yollara tekrar girmemek için

### 4.1 Kaggle: bayat hücre çıktısı gerçek durum sanıldı
Kullanıcı defalarca "yeni" çıktı yapıştırdı ama byte-byte aynıydı (aynı
`nvidia-smi` zaman damgası, aynı `ps` CPU süreleri). **Ders:** Kaggle/Jupyter
sayfa yenilendiğinde son KAYDEDİLMİŞ çıktıyı gösterir. Hücrenin gerçekten
koştuğunu `[*]` göstergesinden doğrula. Tanı için `nvidia-smi` zaman damgası +
log dosyası `mtime` karşılaştır.

### 4.2 `inpaint_add` klasörü 108 kayıt üretti, 200 bekleniyordu
Hata değil: modül çıktısını aynı klasörde iki `manip_type` etiketine bölüyor
(`inpaint_add` 108 + `inpaint_enlarge` 92 = 200). Panik yapmadan `grep` ile
doğrulandı.

### 4.3 splice/copy_move renk uyumsuzluğu — GERÇEK HATA
`_paste_region` %60–70 olasılıkla `cv2.seamlessClone` kullanıyordu. Poisson
blending **büyük/dokulu yamalarda merkez rengini değiştirmez** (yalnızca
sınırdan sızar); kalan %30–40'ta `except cv2.error: pass` ile **sessizce**
düz alfa harmanlamaya düşüyordu ve bu hiçbir yere loglanmıyordu.
Ölçüm (`diag_splice.py`): splice örneklerinin %31'i dE>25 ("alakasız renk").
**Düzeltme:** harmanlamadan ÖNCE `color_transfer`, SONRA `color_consistency_de`
kabul kapısı. **Sonuç: splice %31→%0, copy_move %10→%0.**

### 4.4 `bg_replace` dikdörtgen maske
`vehicle_region` GrabCut yakınsamazsa merkez %60 dikdörtgenine düşüyor —
planın açıkça yasakladığı "dikdörtgen maske → model kenarları ezberler" tuzağı.
`shape_is_rectangular` kapısı eklendi.

### 4.5 `bg_replace` — iki tur yanlış teşhis, sonunda doğru kök neden
- **1. tur:** dikdörtgenlik sanıldı. `diag_bg_shape.py` ile ölçüldü:
  **korelasyon yok** (en kötü 20 örneğin ortalama extent'i 0.678, genelin
  ortalaması 0.675). Hipotez reddedildi.
- **Kendi teşhis script'imde hata:** bg_replace maskesi ARKA PLAN olduğu için
  `RETR_EXTERNAL` konturu hep "tüm resim" veriyordu → extent daima ~0.998.
  `invert=True` ile maske ters çevrilip gövde geri elde edilerek düzeltildi.
  **Ders: teşhis aracının kendisi de doğrulanmalı.**
- **2. tur:** ton uyumsuzluğu. Sabit 21×21 Gauss yalnızca GEOMETRİK geçişi
  yumuşatır, RENK/IŞIK uyumsuzluğuna dokunmaz. Kısmi `color_transfer` (0.5) +
  boyutla ölçeklenen çekirdek + artış kapısı → **artış 5.6 → 0.8**.
- **3. tur (kolajda GÖZLE bulundu):** GrabCut istisna atmadan "başarıyla"
  yakınsıyor ama görüntünün kabaca yarısını düz olmayan bir sınırla "araç"
  sanıyor — `shape_is_rectangular` bunu yakalamaz. **Düzeltme:** CarDD'nin
  kendi hasar maskesi GrabCut'a tohum olarak veriliyor (`hint_mask`).
  **Bu düzeltme yazıldı ve teslim edildi ama VERİ YENİDEN ÜRETİLMEDİ.**

### 4.6 ⭐ GEOMETRİ SIZINTISI — en kritik bulgu, düzeltmesi HENÜZ KOŞULMADI

`scripts/check_leakage.py` ile ölçüldü:

| Özellik | Bulgu |
|---|---|
| Uzun kenar | gerçek/manipüle `{1000, 4032}` · sentetik `{512,640,768,1024,1152}` → **KESİŞİM BOŞ** |
| Genişlik tek başına | AUC **0.861** (aggressive profilinde **0.958**) |
| En-boy oranı | AUC **0.803** |
| Yön | gerçek %91 yatay · sentetik %44 dikey + %16 kare |
| Task B (real vs manip) | AUC **0.506** → **temiz**, sorun yalnız Task A'da |

**`uzun kenar == 1000` tek satırlık kuralı gerçeği sentetikten %100 ayırıyordu.**
AUC 0.861 durumu hafife gösteriyordu — bu yüzden `check_leakage.py` AUC'nin
yanında ayrıca "kesişim boş mu" testi yapar.

Bu, `experiments/E1_shortcut` tablosundaki Task A `meta` AUC 0.976–0.988
sonucunun kök nedenidir ve W3.md'de "açık risk" diye not edilmişti ama
düzeltilmemişti.

**Düzeltme (kod yazıldı, test edildi, diske kondu — AMA ÇALIŞTIRILMADI):**
`src/data/launder.py` içinde her profilin ilk adımı artık:
merkezden **kare kırpma** + **448×448**'e indirme.

- **Neden kare, 4:3 değil:** 4:3 dikey görüntüleri dikey bırakır, sentetiklerin
  %44'ü dikey. Ayrıca CLIP zaten kısa kenarı ölçekleyip merkezden kare kırpıyor
  → E3 için bilgi kaybı ≈ 0.
- **Neden 448, 512 değil:** sentetik havuzun en küçük uzun kenarı 512. Hedef
  512'nin ALTINDA olunca hiçbir görüntü upscale edilmez (upscale yalnız
  sentetiklerde interpolasyon izi bırakır — kapatılan sızıntının yerine yenisi
  gelirdi) ve hiçbiri dokunulmadan geçmez. 448 = CLIP girdisinin (224) 2 katı.
- **Bedeli:** medyan %33 kenar kaybı. `apply_laundering.py` artık kare kırpmada
  **tamamen kaybolan veya yarıya inen maskeleri** raporluyor.

Doğrulandı (sentetik testle): 1000×667, 512×683 dikey, 512 kare ve 4032×3024
girdilerinin hepsi her profilde aynı boyuta çıkıyor, maskeler hizalı kalıyor,
JPEG kalite farkı korunuyor (76KB → 12KB).

### 4.7 Dondurulmuş test seti BOZUK
`test_manifest_frozen.parquet` = 755 satır; güncel manifest test bölümü = 650.
Ortak 608, kaybolan **147**, yeni 42. `test_manifest_frozen.sha256` dosyası **BOŞ**.
Eski hash: `d0e2cb375c3d9602a132dc068fe2fecd5f6850e4a5736c5b305245942ef6f6cc`.

M3 yeniden üretildiği için kaçınılmazdı. **Henüz hiçbir model eğitilmediği
için zararsız** — planın "test setine bakarak model seçme" yasağı ancak sonuç
görüldükten sonra işler. Ama ilk E3 koşusundan **ÖNCE** yeniden dondurulmalı.

### 4.8 Süreç hatası — tekrarlanmaması gereken
İki gün, hedefini **%130 ile aşan tek katmanın** (M3, 520/400) görsel
kalitesine harcandı. Bu sırada S/M1/M2 %25–30'da, geometri sızıntısı açık ve
E3 sonucu sıfırdı. Yapılan düzeltmeler gerçek iyileştirmelerdi ama **en az
kritik katmandaydı ve zamanlaması yanlıştı.**
**Kural: bir sonraki işi seçerken önce planın P0 listesine bak.**

---

## 5. Bir sonraki adım

### 5.1 HEMEN — Adım 1'i tamamla (kod hazır, sadece koşulacak)

```powershell
python scripts\apply_laundering.py --overwrite
python scripts\check_leakage.py
```

- Birincisi ~17.000 laundered kopyayı 448×448 olarak yeniden üretir (sürer).
- İkincisinin **"DURUM: YEŞİL"** demesi beklenir. Sızıntı kalırsa hata kodu
  döner ve devam edilmez.
- `apply_laundering` çıktısındaki **MASKE KORUNUMU** bölümü kontrol edilmeli:
  "tamamen kaybolan maske: 0" değilse o kayıtlar manifest'ten düşülmeli.

> **NOT:** M3 verisi, `hint_mask` düzeltmesi (4.5, 3. tur) uygulanmadan önce
> üretilmiş halde duruyor. İstenirse şu komutla yeniden üretilebilir; ancak
> bu P0 değil, geometri sızıntısı ve E3 önceliklidir:
> ```powershell
> Remove-Item -Recurse -Force data\raw\manipulated\classic
> python -m src.data.generators.classic_manip --manifest data\processed\manifest_v1.parquet --n 520
> ```

### 5.2 Adım 2 — test setini yeniden dondur

```powershell
python scripts\build_manifest_v2.py --freeze-test
```
Yeni sha256'yı `test_manifest_frozen.sha256`'ya yaz, `docs/dataset_card.md` ve
`docs/weekly/W3.md`'yi güncelle (neden yeniden donduruldu: M3 yeniden üretimi +
geometri normalizasyonu, hiçbir model eğitilmeden önce).

### 5.3 Adım 3 — CLIP embedding çıkarımı (Kaggle T4)

`notebooks/W3_kaggle_clip_embed.ipynb` hazır. İçindeki **"BÜTÜNLÜK KAPISI"**
hücresi yerel hash ile karşılaştırma yapıyor — **yeni sha256 ile güncellenmeli**,
yoksa "TUTMUYOR" der ve durur.

### 5.4 Adım 4 — Hafta 3'ün asıl işi

Aynı embedding cache'inden dakikalar içinde koşar:

| Deney | İçerik |
|---|---|
| **E3** | CLIP ViT-L/14 frozen + LogisticRegression — **ANA BASELINE** |
| **E4** | `grip-unina/ClipBased-SyntheticImageDetection` zero-shot — plan "en ilginç sonuç" diyor |
| **E5** | E3 + laundering augmentation |
| **E6** | generator-disjoint (**sd_turbo held-out**, FLUX yerine) |

**Her biri 3 seed ile koşulmalı, ortalama ± std raporlanmalı.**

### 5.5 Adım 5 — Hafta 3 teslimini kapat

- 4–6 deneyin karşılaştırma tablosu (satır: senaryo · sütun: clean/whatsapp/aggressive)
- ROC eğrileri tek grafikte
- Kalibrasyon eğrisi (reliability diagram)
- Seçilmiş model + kaydedilmiş threshold (`probe.joblib`)
- `docs/weekly/W3.md`'ye **"Bulgu: X"** cümlesi

### 5.6 Kapsam kararları — gerçekçi olmak için

| Karar | Gerekçe |
|---|---|
| Veri hedefini **4.800 → ~2.000**'e çek | 1.630 görüntü daha üretmek 2+ tam Kaggle oturumu. Planın P0'ı hacim değil, doğru split + CLIP baseline. |
| **17 deney → ~8**: E0(var), E3, E4, E5, E6, E7, E12, E16 | Planın kendi P0+P1 listesi zaten bu |
| **E10 (IMDL fine-tune) KESİLDİ** | Plan zaten "P2 = kesilebilir" diyor |
| MLflow geriye dönük kurulmayacak | `results.json` + `NOTES.md` sürdürülüp sapma dürüstçe raporlanacak |
| Hafta 6'da önce **Gradio**, Docker/FastAPI zaman kalırsa | Plan: "Gradio'yu tercih et, 100 satırda biter" |

### 5.7 Yeni asistan için ilk komut

Bağlamı doğrulamak için önce şunu koştur — hiçbir şeyi değiştirmez:

```powershell
python scripts\check_leakage.py --manifest data\processed\manifest_v2_laundered.parquet
```

Çıktı **KIRMIZI** ise 5.1 henüz yapılmamış demektir; oradan başla.
**YEŞİL** ise 5.2'ye geç.
