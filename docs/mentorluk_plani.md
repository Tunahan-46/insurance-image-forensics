# AI Image Forensics for Insurance Claims
## Mentorluk Dokümanı ve 6 Haftalık Ar-Ge Yol Haritası

**Hazırlayan:** Kıdemli CV / AI Ar-Ge mentörü
**Hedef kitle:** Bilgisayar Mühendisliği 4. sınıf, Türkiye Sigorta AI stajyeri
**Süre:** 4–6 hafta
**Çıktı:** Çalışan MVP + ölçülmüş sonuçlar + teknik dokümantasyon + yönetici sunumu

---

# 0. Mentor Notu — Bunu Okumadan Başlama

Sana en başta üç sert şey söyleyeceğim. Bu üçünü kabul edersen proje biter, etmezsen 6 hafta sonunda elinde güzel bir Jupyter notebook ve hiçbir şey olur.

### 0.1. Sen yeni bir model icat etmiyorsun. Sen bir **sistem entegratörü** ve **deneyci**sin.

Bu alanda (AI-generated image detection + image manipulation localization) 2023'ten beri çok güçlü, açık kaynak, ön-eğitimli modeller var. Senin işin bunları:
1. Sigorta hasar fotoğrafı domain'ine getirmek,
2. Bu domain'de **gerçekten ne kadar çalıştıklarını ölçmek**,
3. Zayıf oldukları yerleri göstermek,
4. Bir risk skoruna dönüştürüp API'lemek.

Yeni mimari denemesi 6 haftalık projenin katili. Literatürdeki en güçlü baseline'ı al, domain'e uyarla, dürüstçe ölç. Bu bir Ar-Ge katkısıdır — üstelik şirket için "yeni bir loss fonksiyonu"ndan çok daha değerlidir.

### 0.2. Problemi ikiye böl. Bunlar **iki farklı problem**.

| | Task A: Synthetic Detection | Task B: Manipulation Localization |
|---|---|---|
| Soru | Bu görüntü sıfırdan AI ile mi üretildi? | Bu gerçek fotoğrafın **bir bölgesi** değiştirildi mi? Nerede? |
| Çıktı | Görüntü seviyesi skor | Piksel seviyesi maske + görüntü skoru |
| Literatür | UniversalFakeDetect / CLIP-based detection | IMDL (Image Manipulation Detection & Localization) |
| Model | CLIP ViT + linear probe, Corvi/Grag2021 | TruFor, IML-ViT, MVSS-Net, Mesorch |
| Veri | real vs. SD/SDXL/FLUX çıktıları | manipüle görüntü + **ground-truth maske** |
| Zorluk | Görülmemiş generator'a genelleme | Yeniden sıkıştırma sonrası iz kaybı |

Sigorta açısından **Task B daha kritiktir.** Çünkü tipik dolandırıcı sıfırdan bir kaza fotoğrafı üretmez — kendi arabasının **gerçek** fotoğrafını çeker, üstüne inpainting ile çizik/göçük ekler. Görüntünün %95'i gerçek. Global detector bu görüntüyü "gerçek" der. Bu senin sistemindeki en büyük risk ve aynı zamanda projenin en değerli bulgusu olacak.

**Ama Task A daha kolay ve daha hızlı sonuç verir.** Yani: Task A ile başla (haftalar 2–3), Task B'ye geç (haftalar 4–5), ikisini birleştir (hafta 5–6).

### 0.3. Kapsamı daralt. Bugün.

"Sigorta hasar fotoğrafı" çok geniş. Şunu seç ve savun:

> **Kapsam: Kasko/trafik branşında, müşteri tarafından mobil cihazla gönderilen araç dış hasar fotoğrafları (çizik, göçük, cam kırığı, far kırığı, çarpma hasarı).**

Kapsam dışı (sunumda açıkça belirt, bu bir zayıflık değil bilimsel disiplindir):
- Konut/işyeri hasarı, sağlık belgesi, fatura/ekspertiz raporu görselleri (OCR + doküman forensics ayrı bir proje)
- Video / canlı çekim doğrulama
- Yüz deepfake

Neden? Çünkü tek bir görsel domain'de (araç dış yüzeyi) veri üretebilir, tutarlı ölçüm yapabilir ve "domain-specific detector domain-agnostic detector'dan iyi mi?" gibi **gerçek bir araştırma sorusu** sorabilirsin.

---

# 1. Proje Stratejisi

## 1.1. Bu problem sigorta için neden şu anda önemli?

Sektör verisiyle konuş — sunumda ilk slaytın bu olacak:

- Verisk'in Mart 2026 tarihli *State of Insurance Fraud* çalışmasına göre sigortacıların **%98'i** AI destekli düzenleme araçlarının dijital dolandırıcılığı büyüttüğünü söylüyor; **%99'u** halihazırda manipüle edilmiş veya AI ile değiştirilmiş bir belge/görsel ile karşılaştığını belirtiyor. Buna karşılık sadece **%32'si** deepfake tespiti konusunda kendini güvende hissediyor. Tüketici tarafında Gen Z'nin **%55'i** bir hasar fotoğrafını veya belgesini düzenlemeyi "değerlendirebileceğini" söylüyor.
- SAS'ın Mayıs 2026 açıklamasına göre ABD'de sigorta dolandırıcılığının toplam maliyeti yıllık ~308,6 milyar dolar ve mülk-kaza branşındaki hasarların yaklaşık 10'da 1'i dolandırıcılık içeriyor. Sentetik medya bu payın hızla büyüyen bir bileşeni.
- Hukuk cephesinde de gündemde: Debevoise'ın Ocak 2026 tarihli analizi, sigortacıların hasar ödemelerini doğrudan gönderilen görsellere dayandırmasının yeni bir saldırı yüzeyi yarattığını vurguluyor.

**Türkiye Sigorta bağlamında argümanın:**
Dijital hasar ihbarı (mobil app, WhatsApp, web) yaygınlaştıkça, eksper fiziksel olarak aracı görmeden ödeme kararı verilen "fast-track / hızlı hasar" akışları artıyor. Bu akışlar tam olarak sentetik görsel saldırısının hedefidir. Küçük tutarlı hasarlarda (örn. tek panel çizik) manuel ekspertiz maliyeti hasar tutarını aştığı için otomatik onay veriliyor — dolandırıcı bunu bilir.

> **Sunumda kullanacağın tek cümlelik değer önermesi:**
> "Hızlı hasar akışında ekspertiz maliyetini artırmadan, görsel kanıtın güvenilirliğini otomatik olarak skorlayan bir ön-eleme (triage) katmanı."

## 1.2. Neden gerçek hayatta zor? (Bunu bilmen seni stajyerden ayırır)

Bunlar sunumun "Challenges" slaytı ve aynı zamanda deney tasarımının gerekçesi.

**1) Sıkıştırma laundering'i (en büyük problem).**
Forensic izler (JPEG kuantizasyon izleri, sensör gürültüsü, yüksek frekanslı diffusion artifact'ları) çok kırılgandır. Müşteri fotoğrafı WhatsApp'tan gönderdiğinde görüntü yeniden boyutlandırılır ve yeniden JPEG'lenir. Bu tek işlem, laboratuvarda %99 doğruluk veren birçok dedektörü %60'lara düşürür. NTIRE 2026'nın "Robust AI-Generated Image Detection in the Wild" yarışması tam olarak bu problem üzerine kurulu: pratikte görüntüler dedektöre ulaşmadan önce kırpılıyor, yeniden boyutlandırılıyor, yeniden sıkıştırılıyor ve bulanıklaştırılıyor; bu işlemler performansı ciddi biçimde bozuyor.

> **Deneysel karşılığı:** Her modelin performansını "temiz" ve "laundered" test setinde ayrı ayrı raporlayacaksın. Bu senin en çarpıcı grafiğin olacak.

**2) Open-set / generator drift.**
Modelini SD 1.5 üzerinde eğitirsin, dolandırıcı FLUX veya Nano Banana benzeri güncel bir araç kullanır. Klasik binary classifier'ların bilinen zayıflığı budur: "real" sınıfı bir çöp kutusuna dönüşür ve görülmemiş generator'ın çıktısı oraya düşer. CLIP tabanlı yaklaşımların (Ojha vd., CVPR 2023) popüler olmasının sebebi tam olarak budur — real/fake ayrımı için özel eğitilmemiş bir feature space kullanmak, görülmemiş generator'lara genellemede belirgin biçimde daha iyi sonuç veriyor.

> **Deneysel karşılığı:** **Generator-disjoint split.** Eğitimde SD1.5 + SDXL, testte FLUX + SD3 (veya erişebildiğin farklı bir aile). Aynı generator'da test etmek kendini kandırmaktır.

**3) Lokal manipülasyon, global dedektörü atlatır.**
Şubat 2026 tarihli bir çalışma (*AI-Generated Image Detectors Overrely on Global Artifacts*) tam olarak bunu gösteriyor: dedektörler görüntünün geneline yayılmış izlere aşırı bağımlı; inpainting ile yapılan bölgesel değişikliklerde başarısız oluyorlar. Sigorta senaryosunun ana saldırısı da bu.

**4) Ground truth yokluğu.**
Elinde "bu hasar dosyası sahteydi" diye etiketlenmiş gerçek bir veri seti yok ve olmayacak. Bu yüzden **sentetik veri üretimi projenin ana mühendislik işidir**, yan iş değil. Buna gerçek zaman ayır (Hafta 2 tamamen bu).

**5) Asimetrik maliyet ve yasal risk.**
Gerçek bir müşteriyi "sahtekar" diye işaretlemenin maliyeti sadece finansal değil: itibar, şikayet, SEDDK/TSB tarafında regülasyon riski, KVKK. Bu yüzden sistem **karar verici değil, triage** olmalı. Bunu mimarinin merkezine koy (bkz. 1.4).

**6) Domain'in kendisi zor.**
Araç yüzeyi = parlak, yansımalı, düşük dokulu. Gerçek çizikler ince ve yüksek frekanslı; forensic dedektörler için "manipülasyon kenarı" ile "gerçek çizik kenarı" görsel olarak benzeyebilir. Yani bu domain'de **false positive doğal olarak yüksek** olacak. Bunu ölçüp raporlaman bir başarısızlık değil, projenin bulgusudur.

## 1.3. Yaklaşımların değerlendirilmesi

| Yaklaşım | Ne yapar | Artı | Eksi | Sigorta uygunluğu | Karar |
|---|---|---|---|---|---|
| **Metadata / EXIF analizi** | EXIF, cihaz, yazılım imzası, C2PA tutarlılığı | Neredeyse bedava, çok hızlı, açıklanabilir | Kolay silinir/sahtelenir; WhatsApp zaten siler | ⭐⭐⭐⭐ Tek başına yetmez ama **triage'ın ilk katmanı**; "EXIF yok + şüpheli skor" güçlü kombinasyon | **AL** (Hafta 1) |
| **AI-generated image detection (global)** | Görüntü tümüyle sentetik mi | Olgun literatür, güçlü pretrained model | Lokal edit'te kör; generator drift | ⭐⭐⭐⭐ Ana bileşen | **AL** (Hafta 3) |
| **Image manipulation localization (IMDL)** | Değiştirilen bölgeyi segmentler | Sigorta senaryosuna birebir; açıklanabilir maske | FP yüksek; sıkıştırmaya duyarlı; pretrained modelin domain'i farklı | ⭐⭐⭐⭐⭐ **Projenin kalbi** | **AL** (Hafta 4) |
| **Klasik forensics (ELA, gürültü, DCT/FFT, CFA)** | El yapımı izler | Ucuz, sezgisel, güzel görsel | ELA modern editlerde bilimsel olarak zayıf; tek başına güvenilmez | ⭐⭐ Model feature'ı ve **demo görselleştirmesi** olarak değerli, karar mekanizması olarak değil | **SINIRLI AL** |
| **Semantic / physical plausibility** | Gölge-ışık tutarlılığı, hasar-araç uyumu, VLM ile mantık kontrolü | Sıkıştırmadan etkilenmez | Zor, subjektif | ⭐⭐⭐ 6. hafta "future work" veya küçük VLM denemesi | **OPSİYONEL** |
| **Ensemble / evidence fusion** | Sinyalleri birleştirir | Tek dedektörün kırılganlığını dağıtır; kalibre skor | Kalibrasyon gerekir | ⭐⭐⭐⭐⭐ | **AL** (Hafta 5) |
| **Sıfırdan yeni mimari** | — | — | 6 haftada bitmez | ⛔ | **ALMA** |

> **Sonuç: Sigorta fotoğraflarında en uygulanabilir yaklaşım tek bir model değil, kalibre edilmiş çok-sinyalli bir kanıt füzyonudur (multi-signal evidence fusion).** Bu, sektördeki ticari çözümlerin de yaklaşımıdır — SAS'ın Mayıs 2026'da duyurduğu sigorta çözümü de computer vision, OCR ve LLM muhakemesini birleştiren çok-sinyalli bir hattı, kalibre bir risk skoruyla otomatik onay / insana yükseltme / red kararlarına bağlıyor. Senin mimarin bu felsefeyle aynı çizgide olacak; bu, sunumda güçlü bir konumlandırma.

## 1.4. Sistem mimarisi (üst seviye)

```
                       ┌──────────────────────────┐
                       │  Hasar fotoğrafı (input) │
                       └────────────┬─────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
   ┌───────────────────┐  ┌──────────────────┐  ┌────────────────────┐
   │ L1: METADATA      │  │ L2: SYNTHETIC    │  │ L3: MANIPULATION   │
   │  • EXIF varlığı   │  │     DETECTION    │  │     LOCALIZATION   │
   │  • Cihaz/yazılım  │  │  • CLIP ViT-L    │  │  • TruFor / IMLViT │
   │  • JPEG quant tbl │  │    + linear probe│  │  • + tampering mask│
   │  • C2PA manifest  │  │  • Corvi ResNet  │  │  • max/mean skor   │
   │  → p_meta, flags  │  │  → p_synthetic   │  │  → p_manip, mask   │
   └─────────┬─────────┘  └────────┬─────────┘  └─────────┬──────────┘
             │                     │                      │
             │        ┌────────────┴──────────┐           │
             │        │ L4: KLASİK FORENSICS  │           │
             │        │  ELA / noise / FFT    │           │
             │        │  (feature + görsel)   │           │
             │        └────────────┬──────────┘           │
             └─────────────────────┼──────────────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │  L5: FUSION & CALIBRATION    │
                    │  Logistic Regression /       │
                    │  Gradient Boosting           │
                    │  + Platt / Isotonic scaling  │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │  L6: DECISION & EXPLANATION  │
                    │  risk_score ∈ [0,1]          │
                    │  band: DÜŞÜK / ORTA / YÜKSEK │
                    │  evidence[]: hangi sinyal    │
                    │  heatmap + Grad-CAM overlay  │
                    └──────────────┬───────────────┘
                                   ▼
              OTOMATİK ONAY  │  EKSPER İNCELEMESİ  │  DERİN İNCELEME
```

**Tasarımın üç kritik ilkesi (sunumda vurgula):**

1. **Karar değil triage.** Sistem "sahte" demez. Bir risk bandı üretir ve dosyayı bir kuyruğa yönlendirir. Nihai karar insanındır.
2. **Kanıt üretir, skor değil sadece.** Her yüksek skor bir gerekçe listesi ve bir ısı haritası ile gelir. Eksper "neden" sorusunun cevabını görür.
3. **Modüler ve değiştirilebilir.** Her katman bağımsız bir Python modülü ve bağımsız ölçülür. 6 ay sonra L2 modeli eskidiğinde sadece o kutu değişir.

---

# 2. 6 Haftalık Yol Haritası

> **4 haftalık sıkıştırılmış versiyon:** Hafta 1 → Hafta 1, Hafta 2 → Hafta 2, Hafta 3+4 birleşik → Hafta 3, Hafta 5+6 birleşik → Hafta 4. Bu durumda L4 (klasik forensics) ve robustness çalışmasının bir kısmını "future work"e at, kalanı koru.

Her hafta **Cuma günü** bir "demo + kısa rapor" ile bitecek. Bu senin ritmin. Rapor 1 sayfa: ne yaptım, hangi sayı çıktı, ne öğrendim, gelecek hafta ne yapacağım. Bunu `docs/weekly/W1.md` olarak repo'ya commit'le. 6 hafta sonunda teknik dokümantasyonun %60'ı zaten yazılmış olur.

---

## HAFTA 1 — Problem çerçeveleme, veri iskeleti, ölçüm altyapısı

**Amaç:** Hafta sonunda elinde ölçüm yapabilen bir iskelet ve *bir* çalışan zayıf baseline olacak. Model kalitesi umurunda değil; **pipeline'ın uçtan uca dönmesi** önemli.

### Öğrenilecek konular
- IMDL vs. synthetic detection ayrımı, terminoloji (splicing, copy-move, inpainting, removal, outpainting)
- JPEG sıkıştırma temelleri: DCT, kuantizasyon tablosu, double compression kavramı
- EXIF yapısı, C2PA / Content Credentials nedir
- Değerlendirme metrikleri: ROC-AUC, PR-AUC, **sabit FPR'de TPR**, kalibrasyon

### Okunacaklar (yüzeysel oku, 1 gün, not çıkar)
1. Ojha, Li, Lee — *Towards Universal Fake Image Detectors that Generalize Across Generative Models* (CVPR 2023). **En kritik makale. Detaylı oku.**
2. Cozzolino vd. — *Raising the Bar of AI-generated Image Detection with CLIP* (GRIP-UNINA). Repo: `grip-unina/ClipBased-SyntheticImageDetection`
3. Guillaro vd. — *TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization* (CVPR 2023)
4. Ma vd. — *IML-ViT: Benchmarking Image Manipulation Localization by Vision Transformer*. Repo: `SunnyHaze/IML-ViT`
5. Ma vd. — *IMDL-BenCo* (NeurIPS 2024): IMDL alanının standart benchmark/codebase'i. MantraNet, MVSS-Net, CAT-Net, ObjectFormer, PSCC-Net, NCL-IML, TruFor ve IML-ViT'i tek çatı altında topluyor. **Bunu kullan, kendi eğitim döngünü yazma.**
6. Tarama için: `ant-research/Awesome-AIGC-Image-Video-Detection` listesi

> **Okuma disiplini:** Her makale için `docs/lit/` altında 10 satırlık bir not: problem / yöntem / veri / metrik / bizim işimize yarayan kısım / repo linki + lisans. Hafta 6'da bu notlar sunumun literatür slaytı olur.

### Kodlama
```
insurance-image-forensics/
├── src/
│   ├── data/          # veri üretimi, laundering, split
│   ├── detectors/     # her katman ayrı modül, ortak arayüz
│   ├── fusion/
│   ├── eval/          # metrikler, rapor üretimi
│   └── api/
├── configs/           # YAML deney konfigleri
├── experiments/       # her deney = 1 klasör, sonuçlar JSON
├── notebooks/
├── docs/
└── tests/
```

Bu hafta yazacakların:
- `src/detectors/base.py` — **ortak arayüz.** Bu tasarım kararı projeni kurtaracak:
  ```python
  class Detector(Protocol):
      name: str
      def predict(self, image_path: str) -> DetectorOutput: ...
  # DetectorOutput = {score: float, mask: np.ndarray|None,
  #                   features: dict, meta: dict}
  ```
- `src/data/manifest.py` — Her görüntü için tek satır kayıt: `image_id, path, label, source_type, generator, manipulation_type, mask_path, laundering_profile, split`. **Parquet/CSV manifest, klasör yapısına güvenme.**
- `src/eval/metrics.py` — ROC-AUC, PR-AUC, TPR@FPR=1%, TPR@FPR=5%, ECE, confusion matrix, localization için piksel F1 ve IoU
- `src/eval/report.py` — bir manifest + bir detector alır, standart bir sonuç JSON'u ve grafikler üretir
- `src/detectors/metadata.py` — EXIF okuma (`exifread`/`Pillow`), JPEG kuantizasyon tablosu çıkarma, C2PA manifest kontrolü, kural tabanlı `p_meta`

### Veri (bu hafta minimum)
- **Gerçek:** CarDD (cardd-ustc.github.io) — 4.000 yüksek çözünürlüklü araç hasar fotoğrafı, 6 hasar sınıfında 9.000+ anotasyonlu instance. Sigorta domain'ine en yakın açık veri seti; senin "real" sınıfının omurgası. Ek olarak Roboflow Universe'teki araç hasar setleri ve **kendi çektiğin 50–100 telefon fotoğrafı** (otopark turu — çizik, göçük, çamurluk; bunlar "gerçek EXIF'li gerçek veri" olarak altın değerinde).
- **Sahte (hızlı ve kirli):** 200 adet SD/SDXL üretimi araç hasar görseli. Kalitesi kötü olsun, önemli değil — amaç pipeline testi.

### Deney
**E0 — Sanity baseline.** ResNet-50 (ImageNet pretrained) + son katman fine-tune, 400 real / 200 fake üzerinde. Amaç doğruluk değil; manifest → dataloader → train → eval → rapor JSON zincirinin çalıştığını görmek.

### Beklenen çıktı (Cuma)
- Çalışan repo iskeleti, `pytest` geçen 5–10 test
- E0 sonuç JSON'u + ROC eğrisi PNG
- `docs/weekly/W1.md`
- Literatür notları (6 makale)

---

## HAFTA 2 — Veri üretim fabrikası (projenin en önemli haftası)

**Amaç:** 6 saldırı senaryosunu kapsayan, maskeli, laundering profilli, sızıntısız bölünmüş bir veri seti. Bu hafta iyi geçerse geri kalan 4 hafta kolay; kötü geçerse hiçbir sonucun anlamı olmaz.

### Öğrenilecek konular
- Diffusers kütüphanesi: `StableDiffusionXLPipeline`, `StableDiffusionInpaintPipeline`
- Inpainting ve mask conditioning; strength/guidance parametreleri
- SAM (Segment Anything) ile otomatik maske üretimi
- LaMa / OpenCV `inpaint` ile nesne silme
- Poisson blending (`cv2.seamlessClone`) ile klasik splicing
- JPEG yeniden sıkıştırma zincirleri

### Kodlama — 4 üretici modül

`src/data/generators/`:

1. **`fully_synthetic.py`** — SDXL/SD ile sıfırdan hasar fotoğrafı
2. **`inpaint_add.py`** — Gerçek fotoğrafa **olmayan hasar ekleme** (maske otomatik kaydedilir) ← **en kritik senaryo**
3. **`inpaint_remove.py`** — Var olan hasarı/nesneyi silme (LaMa veya SD inpaint)
4. **`classic_manip.py`** — copy-move, splicing, background replacement (klasik, AI'sız)

Ve bir de **`launder.py`** — bu modül her çıktıya uygulanacak:

| Profil | İşlem | Neyi simüle eder |
|---|---|---|
| `clean` | yok | Laboratuvar koşulu |
| `whatsapp` | resize (uzun kenar 1600) + JPEG q≈75 | WhatsApp gönderimi |
| `screenshot` | resize + PNG→JPEG q≈90 | Ekran görüntüsü alıp gönderme |
| `double_jpeg` | q95 → q70 | Kaydet-düzenle-kaydet |
| `aggressive` | resize 1024 + q60 + hafif blur | En kötü senaryo |

> **Bu tablo senin en iyi deneysel katkın.** Çünkü literatürdeki modellerin çoğu `clean` üzerinde raporlanır, sigorta gerçeği `whatsapp`tır.

### Prompt tasarımı (fully synthetic için)

Prompt'ları elle yazma — **kombinatoryal bir şablon motoru** yaz:

```python
TEMPLATE = ("{quality} photo of a {color} {vehicle}, {damage} on the {panel}, "
            "{setting}, {light}, {angle}, {camera}")

quality  = ["insurance claim", "smartphone", "amateur", "handheld", "dashcam-style"]
color    = ["white","silver","black","red","dark blue","grey"]
vehicle  = ["sedan","hatchback","SUV","pickup truck","minibus","compact car"]
damage   = ["deep scratch","large dent","shattered side window","broken headlight",
            "crumpled bumper","scraped paint","cracked windshield"]
panel    = ["front bumper","rear bumper","driver side door","front fender",
            "hood","tailgate","left quarter panel"]
setting  = ["in a parking lot","on a city street","in a repair shop",
            "on a rainy road","in an apartment garage"]
light    = ["overcast daylight","direct sunlight","fluorescent garage lighting",
            "golden hour","flash at night"]
angle    = ["close-up 45 degree angle","wide shot","slightly tilted framing"]
camera   = ["shot on iPhone","shot on Android phone, slight motion blur",
            "slightly out of focus"]
```

**Prompt tasarımının 4 altın kuralı:**
1. **"Amatör" kelimeleri şart.** "professional photography, 8k, cinematic" yazarsan model stüdyo kalitesinde görsel üretir; dedektörün bunu ayırt etmesi trivial olur ve **%99 accuracy'nin sebebi model değil prompt'un olur.** Bu, projeni geçersiz kılar.
2. **Negative prompt kullan:** `"cinematic, 8k, professional, artstation, illustration, cartoon, render"`.
3. **Çeşitlilik = generalization.** Tek generator, tek seed ailesi, tek çözünürlük → sızıntı.
4. **Her üretimi logla:** prompt, seed, model, scheduler, steps, guidance. Manifest'e yaz. Reproducibility.

### Inpainting senaryosu (Task B verisi) — adım adım

```
CarDD gerçek fotoğrafı
   → SAM ile araç panellerini segmentle (veya elle 30 maske çiz)
   → rastgele bir panel maskesi seç, morfolojik olarak yumuşat
   → SD-inpaint("deep scratch and dent on car door", mask=M)
   → çıktı görüntü + M maskesi kaydet
   → launder profili uygula (maske değişmez, görüntü değişir)
```

**Kritik detay:** Maskeyi tam dikdörtgen yapma. Gerçek dolandırıcı serbest el seçim yapar. Maskeleri düzensiz, yumuşak kenarlı üret (`cv2.GaussianBlur` + threshold veya SAM segment sınırları). Dikdörtgen maske → model kenarları ezberler → sahte yüksek performans.

### Hedef veri seti (gerçekçi rakamlar)

| Sınıf | Adet | Kaynak |
|---|---|---|
| R — Gerçek (temiz) | 2.000 | CarDD + Roboflow + kendi fotoğrafların |
| S — Tam sentetik | 1.200 | SD1.5 / SDXL / FLUX-schnell (400'er) |
| M1 — Inpaint: hasar ekleme | 800 | SD-inpaint, maskeli |
| M2 — Inpaint: nesne/hasar silme | 400 | LaMa + SD-inpaint, maskeli |
| M3 — Klasik (copy-move/splice/bg) | 400 | OpenCV, maskeli |
| **Toplam** | **~4.800** | × 3 laundering profili = ~14.400 değerlendirme örneği |

Bu, tek başına bir öğrencinin 5 günde üretebileceği bir hacimdir. Üretimi gece boyunca batch olarak çalıştır (Colab/Kaggle GPU).

### Split kuralları — **buraya çok dikkat et, projeyi en çok burada batırırlar**

1. **Source-image-disjoint:** Bir CarDD fotoğrafı ve ondan türetilmiş tüm manipülasyonlar **aynı** split'te olmalı. Aksi halde model "bu arabayı tanıyorum" der.
2. **Generator-disjoint (kritik test):** Ana train/val'de SD1.5 + SDXL; **test setinde FLUX** tutulur ve model onu hiç görmez. Bu senin "unseen generator" deneyin.
3. **Laundering:** Train'de karışık profiller (augmentation olarak), test'te her profil **ayrı ayrı raporlanır**.
4. **Test seti dondurulur.** Hafta 2 Cuma günü test setini yaz, hash'le, `test_manifest.parquet` olarak commit'le ve bir daha ellemeyeceksin. Test setine bakarak model seçme = kendini kandırma.

### Beklenen çıktı (Cuma)
- `src/data/` altında 5 çalışan modül
- ~4.800 görüntülü, maskeli, manifest'li veri seti
- `docs/dataset_card.md` — hangi senaryo, kaç adet, nasıl üretildi, lisans notları
- Her senaryodan 3 örnek gösteren bir görsel kolaj (**sunumda kesin kullanacaksın**)
- Dondurulmuş test manifest'i

---

## HAFTA 3 — Synthetic detection (Task A)

**Amaç:** Görüntü-seviyesi AI üretim tespitinde ölçülmüş, karşılaştırmalı, kalibre bir model.

### Öğrenilecek konular
- CLIP mimarisi, feature extraction, linear probing vs. fine-tuning
- Transfer learning stratejileri: frozen backbone, partial unfreeze, LoRA
- Class imbalance, threshold seçimi, kalibrasyon (Platt scaling, isotonic)
- Grad-CAM / attention rollout

### Deneyler

| ID | Model | Eğitim | Neden yapıyoruz | Beklenen (temiz / whatsapp) |
|---|---|---|---|---|
| **E1** | ResNet-50 ImageNet, full fine-tune | Bizim veri | Klasik baseline, üst sınır illüzyonu | AUC 0.95+ / 0.75 |
| **E2** | EfficientNet-B0 / ConvNeXt-T | Bizim veri | Mimari duyarlılığı | E1'e yakın |
| **E3** | **CLIP ViT-L/14 frozen + Logistic Regression** | Sadece linear head | **Ana baseline.** Ojha vd. paradigması | AUC 0.93+ / 0.85+ |
| **E4** | Hazır pretrained: `grip-unina/ClipBased-SyntheticImageDetection` | Zero-shot (eğitim yok) | Literatür SOTA'sı bizim domain'de ne yapıyor? | ? — **en ilginç sonuç** |
| **E5** | E3 + laundering augmentation | Bizim veri, JPEG/resize augment | Robustness kazanımı ölçümü | whatsapp AUC +0.05–0.15 |
| **E6** | E3, generator-disjoint (FLUX unseen) | SD1.5+SDXL train | Genelleme sınırı | E1'e göre belirgin üstünlük bekleniyor |

**E4 özellikle önemli:** Hiç eğitim yapmadan, hazır bir SOTA modeli indirip senin sigorta veri setinde ölçmek 1 günlük iştir ve muhtemelen en güçlü tek sonucun olur. "Literatür modeli laboratuvarda %99, bizim domain'de X%" cümlesi bir Ar-Ge bulgusudur.

### Eğitim reçetesi (E3 için — bu kadar basit)
```
1. CLIP ViT-L/14 yükle, tüm ağırlıkları dondur
2. Tüm görüntüler için 768-d embedding çıkar, .npy olarak cache'le (bir kez!)
3. sklearn LogisticRegression(C=?, class_weight='balanced') ile grid search
4. Val setinde threshold seç (TPR@FPR=1% maksimize)
5. Platt scaling ile kalibre et
```
Cache'leme sayesinde tüm hiperparametre araması CPU'da dakikalar sürer. GPU'yu sadece embedding çıkarma için kullanırsın. **Bu, kısıtlı kaynakla çalışmanın profesyonel yolu.**

### Kodlama
- `src/detectors/clip_probe.py`, `src/detectors/cnn_baseline.py`
- `src/features/clip_embed.py` (cache mekanizmalı)
- `src/eval/calibration.py`
- `src/explain/gradcam.py`

### Beklenen çıktı (Cuma)
- 6 deneyin karşılaştırmalı tablosu (temiz / whatsapp / aggressive sütunlarıyla)
- ROC eğrileri tek grafikte
- Kalibrasyon eğrisi (reliability diagram)
- Seçilmiş model + kaydedilmiş threshold
- `docs/weekly/W3.md` + **"Bulgu: X"** cümlesi

---

## HAFTA 4 — Manipulation localization (Task B)

**Amaç:** Manipüle bölgeleri gösteren maske üreten bir katman ve onun dürüst ölçümü.

### Öğrenilecek konular
- Semantic segmentation metrikleri: piksel F1, IoU, MCC; permutation problemi
- IMDL modellerinin girdi/çıktı sözleşmeleri
- Noiseprint / SRM filtreleri / high-pass residual mantığı
- Görüntü-seviyesi skoru maskeden türetme (max, mean, top-k mean)

### Deneyler

| ID | Yöntem | Eğitim | Amaç |
|---|---|---|---|
| **E7** | TruFor (pretrained, zero-shot) | Yok | Ana localization baseline |
| **E8** | IML-ViT (pretrained, CASIAv2/CAT-Net protokolü) | Yok | İkinci baseline, karşılaştırma |
| **E9** | MVSS-Net veya CAT-Net (IMDL-BenCo üzerinden) | Yok | Üçüncü görüş |
| **E10** | E7/E8'den en iyisi + **bizim veri ile fine-tune** | 500–800 maskeli örnek | Domain adaptasyonu kazancı |
| **E11** | Maske → görüntü skoru dönüşümü | — | Hangi agregasyon (max / top-5% mean) en iyi AUC verir |

**E10 stratejisi:** Sıfırdan eğitim yok. Encoder'ı dondur, decoder'ı düşük LR ile 10–20 epoch eğit. Elinde maskeli 1.600 örnek var; bu fine-tune için yeterli, sıfırdan eğitim için değil.

**Ölçüm dürüstlüğü:** Localization'da iki sayıyı ayrı raporla:
- **Detection AUC** (bu görüntüde manipülasyon var mı?)
- **Localization F1/IoU**, yalnızca *gerçekten manipüle edilmiş* görüntüler üzerinde

Bunları karıştırmak literatürde de sık yapılan bir hatadır; ayırman kaliteni gösterir.

**Ayrıca ölç:** Gerçek (temiz) görüntülerde model ne kadar alan işaretliyor? Bu senin **false positive area rate**'in. Araç yüzeyinde parlama ve gerçek çizikler yüzünden yüksek çıkacak — bu bulguyu raporla, saklama.

### Kodlama
- `src/detectors/trufor_wrapper.py`, `src/detectors/imlvit_wrapper.py` (ortak `Detector` arayüzüne uydur)
- `src/eval/localization_metrics.py`
- `src/explain/overlay.py` — orijinal + maske + ısı haritası yan yana görsel

### Beklenen çıktı (Cuma)
- Localization karşılaştırma tablosu (senaryo × model × laundering)
- **Kalitatif galeri:** 12 örnek — 4 başarı, 4 başarısızlık, 4 false positive. Bu galeri sunumun en ikna edici slaytıdır.
- Fine-tune kazancı (öncesi/sonrası)

---

## HAFTA 5 — Füzyon, kalibrasyon, robustness, açıklanabilirlik

**Amaç:** Dağınık sinyalleri tek bir savunulabilir risk skoruna çevirmek.

### Kodlama
- `src/fusion/feature_builder.py` — her görüntü için sinyal vektörü:
  ```
  [p_synthetic_clip, p_synthetic_cnn, p_manip_trufor, mask_area_ratio,
   mask_max_score, mask_compactness, exif_present, exif_software_flag,
   jpeg_quality_est, double_jpeg_flag, c2pa_present, ela_energy, fft_peak_score]
  ```
- `src/fusion/model.py` — Logistic Regression (yorumlanabilir, katsayılar = kanıt ağırlıkları) **ve** LightGBM (performans). İkisini karşılaştır, LR'yi tercih et eğer fark küçükse — açıklanabilirlik sigortada teknik performanstan önemli.
- `src/fusion/calibrate.py` — isotonic regression
- `src/explain/evidence.py` — insan-okunur gerekçe üretimi:
  ```
  "YÜKSEK RİSK (0.87)
   • Sol ön çamurlukta 4.2% alanda manipülasyon izi tespit edildi
   • EXIF verisi bulunmuyor (yeniden kaydedilmiş olabilir)
   • Çift JPEG sıkıştırma izi mevcut
   • Sentetik üretim olasılığı düşük (0.12) → tam sentetik değil, bölgesel düzenleme"
  ```

### Deneyler

| ID | İçerik |
|---|---|
| **E12** | Füzyon vs. en iyi tekil model — AUC, TPR@FPR=1% karşılaştırması |
| **E13** | **Ablation:** her sinyali tek tek çıkar, performans düşüşünü ölç → "hangi sinyal ne kadar değerli" tablosu. **Sunumun en teknik slaytı bu.** |
| **E14** | **Robustness matrisi:** 5 laundering profili × 6 senaryo → ısı haritası tablo |
| **E15** | **Adversarial sanity:** çıktıya hafif Gaussian gürültü / bilinçli JPEG düşürme uygulandığında skor ne oluyor? Sistemin ne kadar kolay atlatıldığını dürüstçe ölç |
| **E16** | Operasyonel simülasyon: %1 sahte oranına sahip 10.000 dosyalık varsayımsal akışta, eşik 0.8 iken kaç dosya insana gider, kaç sahte yakalanır, kaç masum işaretlenir |

**E16 yönetici sunumunun can damarıdır.** AUC yöneticiye bir şey söylemez; "günde 200 dosyanın 12'si incelemeye gider, sahtelerin %78'i bu 12'nin içindedir" cümlesi karar aldırır.

### Beklenen çıktı
- Kalibre füzyon modeli (`models/fusion_v1.pkl`)
- Ablation ve robustness tabloları
- Açıklama üretici modül
- Operasyonel simülasyon grafiği

---

## HAFTA 6 — MVP, dokümantasyon, sunum

### Kodlama
- **Backend (FastAPI):** `POST /analyze` (multipart image) → JSON; `GET /health`; `GET /explain/{id}` → overlay PNG
- **Frontend:** Gradio veya Streamlit — yükle, analiz et, skoru ve ısı haritasını gör. **Gradio'yu tercih et**, 100 satırda biter.
- **Docker:** `Dockerfile` + `docker-compose.yml`. Model ağırlıkları volume/HF Hub'dan.
- **Testler:** `pytest` — API sözleşmesi, determinizm (aynı görüntü → aynı skor), sınır durumlar (bozuk dosya, çok büyük görüntü, PNG/HEIC), latency testi

### Dokümantasyon
- `README.md` — problem, mimari şeması, kurulum, örnek çıktı, sonuç tablosu özeti
- `docs/technical_report.md` — 10–15 sayfa: problem, literatür, veri, deneyler, sonuçlar, limitasyonlar, gelecek çalışma
- `docs/model_card.md` — model ne için tasarlandı, ne için tasarlanmadı, bilinen zayıflıklar, etik notlar. **Sigorta gibi regüle bir sektörde bu belge çok iyi karşılanır.**
- `docs/limitations.md` — dürüst zayıflık listesi

### Sunum
Bkz. Bölüm 9.

### Beklenen çıktı
- Çalışan, Docker'da ayağa kalkan demo
- Public/internal GitHub repo
- 15–18 slaytlık sunum + 3 dakikalık canlı demo senaryosu

---

# 3. Teknik Mimari — Yöntemlerin Derinlemesine Değerlendirmesi

Her yöntem için: **nasıl çalışır → artı → eksi → sigorta fotoğraflarında potansiyel → benim kararım.**

## 3.1. Görüntü analizi omurgaları

### CNN (ResNet / EfficientNet / ConvNeXt) + transfer learning
**Nasıl çalışır:** ImageNet'te ön-eğitilmiş bir ağın son katmanları real/fake ikili sınıflandırma için yeniden eğitilir. Ağ, generator'a özgü yüksek frekanslı izleri (upsampling artifact'ları, checkerboard desenleri) öğrenir.
**Artı:** Basit, hızlı, tek GPU'da eğitilir, iyi anlaşılmış.
**Eksi:** Eğitimde gördüğü generator'a aşırı uyum sağlar. Yeni bir generator geldiğinde çöker. "Real" sınıfı bir çöp kutusuna dönüşür — bilinmeyen her şey oraya düşer. JPEG'e duyarlı.
**Sigorta potansiyeli:** ⭐⭐ Baseline olarak zorunlu (karşılaştırma için), üretim modeli olarak riskli.
**Karar:** E1/E2 olarak yap, sonuçları *karşılaştırma referansı* olarak kullan.

### Vision Transformer (ViT) + fine-tune
**Nasıl çalışır:** Görüntü patch'lere bölünür, self-attention ile uzak-mesafe ilişkiler modellenir. Manipülasyon tespitinde avantajı: manipüle bölge ile otantik bölge arasındaki **karşılaştırmayı** doğal olarak yapabilmesi — IML-ViT'in temel argümanı tam olarak budur (yüksek çözünürlük kapasitesi, çok-ölçekli özellik ve manipülasyon kenarı denetimi).
**Artı:** Lokalizasyonda CNN'lerden güçlü, uzun menzilli tutarsızlıkları yakalar.
**Eksi:** Veri açlığı yüksek, sıfırdan eğitmek senin bütçende değil.
**Sigorta potansiyeli:** ⭐⭐⭐⭐ Ama **pretrained olarak** kullan.
**Karar:** IML-ViT'i hazır ağırlıklarla kullan (E8), gerekirse decoder fine-tune (E10).

### CLIP tabanlı yöntemler (frozen backbone + linear probe) — **ana önerim**
**Nasıl çalışır:** CLIP ViT, internet ölçeğinde görüntü-metin çiftleriyle eğitilmiş bir görsel temsil uzayı sunar. Bu uzay real/fake ayrımı için *özel olarak eğitilmemiştir*; işte anahtar nokta bu. Bu uzayda gerçek fotoğraflar ile üretilmiş görüntüler doğal olarak farklı bölgelere düşer. Üzerine küçük bir lineer sınıflandırıcı (veya nearest-neighbor) yeterlidir. Ojha vd.'nin gösterdiği üzere, bu basit yaklaşım görülmemiş diffusion ve autoregressive modellere genellemede eğitilmiş dedektörleri belirgin farkla geçiyor. Cozzolino vd.'nin GRIP-UNINA çalışması bunu bir adım ileri taşıyor: tek bir generator'dan sadece birkaç örnekle bile CLIP tabanlı dedektör, DALL·E 3, Midjourney v5, Firefly gibi ticari araçlar dahil geniş bir yelpazede şaşırtıcı derecede iyi genelleme ve yüksek dayanıklılık gösteriyor.
**Artı:** Eğitim maliyeti neredeyse sıfır (embedding cache'lersen dakikalar). Az veriyle çalışır. Open-set genellemesi güçlü. Embedding'ler füzyon katmanına doğrudan feature olarak girebilir.
**Eksi:** Semantik özelliklere de duyarlı — yani "AI görüntüleri stüdyo estetiğinde olur" gibi bir kestirme yol (shortcut) öğrenebilir. **Bu yüzden Hafta 2'deki prompt disiplinin hayati.** Ayrıca lokal inpainting'te zayıftır (görüntünün %95'i gerçek).
**Sigorta potansiyeli:** ⭐⭐⭐⭐⭐ Task A için en iyi hız/performans dengesi.
**Karar:** **Ana synthetic detection baseline'ın bu olacak (E3/E4).**

### Feature extraction + klasik ML
**Nasıl çalışır:** Derin embedding'ler (CLIP/DINOv2) + el yapımı forensic feature'lar bir tabloya çıkarılır, üzerine LR/GBM eğitilir.
**Artı:** Yorumlanabilir, hızlı iterasyon, küçük veriyle stabil, kalibre etmesi kolay.
**Eksi:** Feature mühendisliği emek ister.
**Sigorta potansiyeli:** ⭐⭐⭐⭐⭐ **Füzyon katmanın (L5) tam olarak budur.** Regüle sektörde katsayıları gösterebilmek büyük avantaj.

## 3.2. Klasik dijital görüntü forensics

### Error Level Analysis (ELA)
**Nasıl çalışır:** Görüntü bilinen bir kalitede yeniden JPEG'lenir, orijinal ile farkı alınır. Farklı sıkıştırma geçmişine sahip bölgelerin farklı "hata seviyesi" göstereceği varsayılır.
**Artı:** 15 satır kod, anında görsel, sezgisel olarak ikna edici.
**Eksi:** **Bilimsel olarak zayıf ve yanıltıcıdır.** Modern diffusion inpainting'de görüntünün tamamı yeniden kodlandığı için ELA hiçbir şey göstermez. Yüksek dokulu bölgeler (araç ızgarası, yazılar) her zaman "parlak" çıkar ve masum görüntülerde sahte alarm üretir. Adli literatürde tek başına delil kabul edilmez.
**Sigorta potansiyeli:** ⭐⭐ **Karar mekanizması olarak KULLANMA.** Füzyona zayıf bir feature (`ela_energy`) olarak ve demoda "klasik yöntemler neden yetmiyor" anlatısı için kullan.
**Karar:** Uygula ama **ablation'da zayıf çıkacağını göster.** Bu, olgunluk işaretidir. Birçok stajyer projesi ELA'yı ana yöntem yapıp batar.

### Noise pattern analysis (PRNU / Noiseprint / SRM residual)
**Nasıl çalışır:** Her kamera sensörü benzersiz bir gürültü deseni (PRNU) bırakır. Öğrenilmiş versiyonu Noiseprint, cihaz/işlem izini çıkarır ve tutarsız bölgeleri açığa çıkarır. TruFor bu fikri modern hale getirir: RGB içeriği ile öğrenilmiş, gürültüye duyarlı bir parmak izini transformer tabanlı bir mimaride birleştirir.
**Artı:** Semantikten bağımsız, gerçekten "fiziksel" bir iz. Lokal manipülasyonda güçlü.
**Eksi:** Yeniden sıkıştırma ve yeniden boyutlandırma izi büyük ölçüde siler. Gerçek PRNU eşleştirmesi için aynı cihazdan çok sayıda referans fotoğraf gerekir (sigortada yok).
**Sigorta potansiyeli:** ⭐⭐⭐⭐ Noiseprint tabanlı modern IMDL modelleri (TruFor) üzerinden dolaylı olarak kullan. Ham PRNU'yu deneme.

### Frequency domain analysis (DCT / FFT / spektral)
**Nasıl çalışır:** Fourier veya DCT spektrumunda, GAN upsampling'in bıraktığı periyodik tepe noktaları veya diffusion denoising'in bıraktığı spektral imzalar aranır.
**Artı:** Ucuz, semantikten bağımsız, güzel görselleştirme.
**Eksi:** Modern diffusion modellerinde GAN kadar belirgin değil; JPEG bu izleri bastırır.
**Sigorta potansiyeli:** ⭐⭐⭐ Füzyon feature'ı (`fft_peak_score`) olarak değerli, tek başına değil.

### JPEG artifact analysis (double compression, quantization table)
**Nasıl çalışır:** Bir JPEG düzenlenip yeniden kaydedildiğinde, DCT katsayı histogramlarında periyodik desenler oluşur. Ayrıca kuantizasyon tablosu üreten yazılımın parmak izidir — iPhone'un tablosu ile Photoshop'un veya Python/PIL'in tablosu farklıdır.
**Artı:** Ucuz, sağlam, çok açıklanabilir. "Bu görüntü bir iPhone tarafından değil, bir Python kütüphanesi tarafından kaydedilmiş" cümlesi eksper için son derece anlamlıdır.
**Eksi:** Sadece "yeniden kaydedilmiş" der, "manipüle edilmiş" demez. WhatsApp da yeniden kaydeder → çok fazla false positive.
**Sigorta potansiyeli:** ⭐⭐⭐⭐ **Yüksek ROI, düşük maliyet.** Ama mutlaka diğer sinyallerle birlikte.

### Metadata / EXIF / C2PA analizi
**Nasıl çalışır:** EXIF varlığı, cihaz modeli, `Software` alanı, GPS, çekim zamanı ile ihbar zamanı tutarlılığı, thumbnail-görüntü tutarsızlığı, C2PA/Content Credentials manifest doğrulaması.
**Artı:** **Milisaniyeler.** Sıfır GPU. Tamamen açıklanabilir. Bazı vakaları anında kapatır (`Software: Adobe Photoshop 26.0` veya bir AI aracının imzası).
**Eksi:** Kolay silinir, kolay sahtelenir. WhatsApp zaten temizler → yokluğu suç kanıtı değildir.
**Sigorta potansiyeli:** ⭐⭐⭐⭐ **İlk katman olarak zorunlu.** Ayrıca operasyonel bir öneri doğurur: "Şirket kendi mobil uygulamasında fotoğrafı in-app çektirsin ve C2PA/imzalı olarak yüklesin" — bu, sunumundaki en değerli iş önerisi olabilir, çünkü problemi tespit etmekten kaçınma seviyesine taşır.

## 3.3. Generative AI'ye özgü analiz

### Diffusion / SD üretim tespiti (DIRE, reconstruction-based)
**Nasıl çalışır:** DIRE gibi yöntemler görüntüyü bir diffusion modeliyle ters çevirip (DDIM inversion) yeniden oluşturur; üretilmiş görüntüler kendi model ailesi tarafından daha düşük hatayla yeniden inşa edilir.
**Artı:** Prensip olarak zarif, diffusion'a özgü.
**Eksi:** **Çok pahalı** (görüntü başına saniyeler), kullanılan diffusion modeline bağımlı, JPEG'e duyarlı.
**Sigorta potansiyeli:** ⭐⭐ Gerçek zamanlı hasar akışında maliyeti karşılamaz.
**Karar:** Literatürde anlat, **uygulama.** Zaman tuzağı.

### GAN-generated image detection (spektral izler)
**Nasıl çalışır:** Transposed convolution'ın bıraktığı checkerboard ve periyodik spektral tepeler.
**Sigorta potansiyeli:** ⭐ 2026'da dolandırıcı GAN kullanmıyor, diffusion tabanlı ticari araç kullanıyor.
**Karar:** Sadece tarihsel bağlam olarak sunumda bir cümle.

### Watermark / provenance (C2PA, SynthID benzeri)
**Nasıl çalışır:** Üretici model çıktıya görünmez bir imza veya kriptografik manifest gömer.
**Artı:** Pozitif tespitte kesinlik yüksek.
**Eksi:** **Yokluğu hiçbir şey kanıtlamaz.** Açık kaynak modeller watermark'sız üretir; ekran görüntüsü ile kolayca kırılır.
**Sigorta potansiyeli:** ⭐⭐⭐ "Varsa kesin bilgi, yoksa bilgi yok" mantığıyla asimetrik bir kural olarak füzyona ekle.

### VLM tabanlı semantik tutarlılık (opsiyonel, Hafta 6 keşfi)
**Nasıl çalışır:** Bir vision-language modeline "Bu fotoğraftaki gölgeler tutarlı mı? Hasarın geometrisi çarpma fiziğiyle uyumlu mu? Yansımalarda anormallik var mı?" diye sorulur.
**Artı:** Sıkıştırmadan tamamen bağımsız — piksel izleri silinse bile fizik yalan söylemez. Sektörün gittiği yön de bu (çok-sinyalli: CV + OCR + LLM muhakemesi).
**Eksi:** Kalibre etmesi zor, halüsinasyon riski, ölçmesi zor.
**Karar:** Zamanın kalırsa 20 örnekte kalitatif bir mini-deney. Sunumda "gelecek çalışma" olarak güçlü durur.

---

# 4. Veri Seti Oluşturma Stratejisi (Detay)

## 4.1. Temel ilke

> **Sen bir veri seti üretmiyorsun, bir *saldırgan modeli* üretiyorsun.**
> Sorman gereken soru "nasıl fake görüntü üretirim" değil, **"Türkiye'de kasko hasarı ile 40.000 TL almaya çalışan biri, elinde telefonu ve ücretsiz bir AI editörü varken tam olarak ne yapar?"**

Cevap genellikle şudur, ve zorluk sırası da budur:

| # | Saldırı | Gerçek hayatta olasılık | Tespit zorluğu |
|---|---|---|---|
| 1 | Gerçek arabaya **olmayan hasar ekleme** (inpaint) | 🔥 Çok yüksek | 🔴 Zor |
| 2 | Var olan hasarı **büyütme/şiddetlendirme** | 🔥 Çok yüksek | 🔴 Çok zor |
| 3 | Başka bir aracın hasar fotoğrafını kullanma (plaka/renk düzenleyerek) | Yüksek | 🟡 Orta |
| 4 | **Nesne silme** (eski hasar, park cezası, tarih damgası) | Orta | 🟡 Orta |
| 5 | **Arka plan değiştirme** (olay yerini uydurma) | Orta | 🟢 Kolay-orta |
| 6 | **Tam sentetik** kaza görüntüsü üretme | Düşük-orta | 🟢 Kolay |

**Dikkat et:** Senin listendeki "tam sentetik üretim" aslında **en kolay tespit edilen ve en az olası** saldırı. Buna rağmen literatürün çoğu bununla ilgili. Bu boşluğu sunumda göstermen, projeni sıradan bir "AI detector" demosundan ayırır.

## 4.2. Veri katmanları

**R — Gerçek (Real) katmanı**

| Kaynak | Adet | Not |
|---|---|---|
| CarDD | ~2.000 | Araç hasarı; dent, scratch, crack, glass shatter, lamp broken, tire flat. Akademik kullanım — lisansı oku ve dokümanda belirt. |
| Roboflow Universe araç hasar setleri | 500–1.000 | Kalite değişken, filtrele |
| Stanford Cars / COCO (araç sınıfı) | 500 | "Hasarsız gerçek araç" negatif örnekleri |
| **Kendi telefon fotoğrafların** | 50–150 | 🔑 **Tam EXIF'li, bilinen cihazlı gerçek veri.** Metadata katmanının doğrulanması için tek gerçek kaynağın. Bir otoparkta 1 saat, 100 fotoğraf. Bunu mutlaka yap. |
| Meslektaşlarından toplanan fotoğraflar | 50–100 | İzin alarak, kişisel veri içermeyen (plaka bulanıklaştır) |

> **KVKK notu:** Gerçek hasar dosyası verisine erişimin olursa bile plaka, yüz, konum verisi anonimleştirilmeden repo'ya girmemeli. Sunumda bu hassasiyeti göstermen çok iyi karşılanır. Varsayılan planın **hiç gerçek müşteri verisi kullanmamaktır**; sentetik veri bu yüzden de doğru seçim.

**S — Tam sentetik katmanı**
- SD 1.5 (hızlı, zayıf), SDXL (kaliteli), FLUX.1-schnell (güncel, **test için ayır**), mümkünse SD3 veya bir API modeli
- Her generator'dan 400 görüntü, çeşitli çözünürlük (512, 768, 1024) ve en-boy oranı
- **Kritik:** Her generator ayrı bir `generator` etiketiyle manifest'e yazılır

**M — Manipülasyon katmanı (maskeli)**

| Alt tip | Yöntem | Maske kaynağı |
|---|---|---|
| M1a: Hasar ekleme | SD-inpaint, prompt="deep scratch and dent" | Üretim maskesi |
| M1b: Hasar büyütme | Mevcut CarDD hasar maskesini genişlet + inpaint | Genişletilmiş maske |
| M2a: Nesne silme | LaMa veya SD-inpaint, prompt="clean car door" | Silinen bölge maskesi |
| M2b: Damga/plaka silme | Aynı | Aynı |
| M3a: Copy-move | Aynı görüntüden bölge kopyala + `cv2.seamlessClone` | Yapıştırma maskesi |
| M3b: Splicing | Başka araçtan hasar bölgesi kes + yapıştır | Yapıştırma maskesi |
| M3c: Arka plan değiştirme | SAM ile araç segmenti + arka planı SD ile yeniden üret | Arka plan maskesi (ters) |
| M4: Bölgesel AI edit | Instruct-pix2pix / SDXL img2img düşük strength, sadece bir panelde | Değişim eşiği maskesi |

**Otomatik maske üretimi (zaman kazandırır):**
```
SAM (segment-anything) → araç maskesi + panel segmentleri
   → rastgele bir segment seç
   → morfolojik erozyon/dilatasyon ile düzensizleştir
   → GaussianBlur + threshold ile yumuşak kenar
   → inpaint pipeline'a besle
```

## 4.3. Prompt tasarımı — negatif örnekler

**❌ Yapma:**
```
"professional photograph of a damaged car, 8k, ultra detailed, cinematic lighting, artstation"
```
Bu bir stok fotoğraf üretir. Dedektörün %99.8 accuracy alır. Sonuç anlamsızdır. Değerlendirmen **model kalitesini değil, prompt naifliğini** ölçmüş olur.

**✅ Yap:**
```
positive: "insurance claim photo, silver hatchback, deep scratch on the driver
           side door, apartment parking garage, fluorescent lighting, close-up
           45 degree angle, shot on Android phone, slight motion blur"
negative: "professional, cinematic, 8k, artstation, illustration, render,
           cartoon, oversaturated, studio lighting, perfect composition"
```

**Kalite kontrolü:** Ürettiğin 1.200 sentetik görüntüden rastgele 100'ünü **kendin gözle incele.** Bariz bozuk olanları (5 tekerlekli araba, eriyen plaka) at. Bir arkadaşına 20 gerçek + 20 sentetik karışık göster, kaçını doğru bilebiliyor sor. **Bu insan baseline'ı sunumun altın slaytıdır:** "İnsan %62 doğrulukla ayırt edebiliyor, sistemimiz %X."

## 4.4. Laundering — kritik ve çoğu kişinin atladığı adım

```python
LAUNDER_PROFILES = {
  "clean":       [],
  "whatsapp":    [resize_long_edge(1600), jpeg(75)],
  "screenshot":  [resize_long_edge(1280), png_roundtrip, jpeg(90)],
  "double_jpeg": [jpeg(95), jpeg(70)],
  "aggressive":  [resize_long_edge(1024), jpeg(60), gaussian_blur(0.5)],
}
```

Her test görüntüsü 5 profilde de değerlendirilir. **Sonuç tablon senaryo × profil matrisidir.** Bu matris, projenin bilimsel ağırlığını taşıyan şeydir — çünkü NTIRE 2026'nın da vurguladığı gibi, gerçek dağıtım koşullarında görüntüler dedektöre ulaşmadan önce zaten kırpılmış, boyutlandırılmış ve yeniden sıkıştırılmış oluyor ve bu işlemler performansı ciddi biçimde düşürüyor.

## 4.5. Split ve veri sızıntısı — 5 tuzak

| # | Tuzak | Nasıl önlenir |
|---|---|---|
| 1 | Aynı kaynak fotoğrafın türevleri farklı split'lerde | `source_image_id` bazlı gruplu split |
| 2 | Real'ler 1024px, fake'ler 512px → model çözünürlüğü öğrenir | Tüm görüntüleri aynı boru hattından geçir, çözünürlük dağılımını eşle |
| 3 | Real'ler JPEG, fake'ler PNG → model formatı öğrenir | **Her şeyi aynı JPEG kalitesinde yeniden kaydet** |
| 4 | Fake'ler daha estetik/temiz → model estetiği öğrenir | Prompt disiplini + negative prompt |
| 5 | Test setine bakarak hiperparametre seçme | Test setini Hafta 2'de dondur, sadece Hafta 5–6'da bir kez aç |

> **Tuzak 2 ve 3 en sinsi olanlardır.** Eğer AUC'un 0.99 çıkıyorsa sevinme — önce bu iki tuzağı kontrol et. Hızlı bir teşhis testi: **görüntüleri 32×32'ye küçültüp aynı modeli eğit.** Hâlâ %95 alıyorsan model forensic iz değil, düşük seviyeli istatistiksel bir kestirme yol öğreniyor demektir. Bu testi mutlaka yap ve sonucunu raporla.

---

# 5. Teknoloji Stack

## Çekirdek
```
Python 3.11
torch, torchvision              # PyTorch ekosistemi
timm                            # ConvNeXt, ViT, EfficientNet — tek satırda pretrained
transformers                    # CLIP, ViT
open_clip_torch                 # CLIP varyantları (ViT-L/14 vb.)
diffusers, accelerate           # SD / SDXL / FLUX, inpainting
```

## Görüntü işleme & forensics
```
opencv-python                   # seamlessClone, inpaint, morfoloji
Pillow                          # JPEG quantization table erişimi (img.quantization)
scikit-image                    # metrik, filtreleme
numpy, scipy                    # FFT/DCT
exifread / piexif               # EXIF
pillow-heif                     # iPhone HEIC desteği (gerçek hayatta lazım!)
```

## ML & değerlendirme
```
scikit-learn                    # LogisticRegression, kalibrasyon, metrikler
lightgbm                        # füzyon alternatifi
pandas, pyarrow                 # manifest (parquet)
matplotlib, seaborn             # grafikler
```

## Deney takibi & kalite
```
mlflow  veya  wandb             # deney kaydı — ZORUNLU, not defteri yeterli değil
hydra-core / OmegaConf          # konfigürasyon
pytest, ruff, black             # test + lint
dvc                             # (opsiyonel) veri versiyonlama
```

## Servis
```
fastapi, uvicorn, python-multipart
gradio                          # demo UI — Streamlit'e tercih et, hızlı
pydantic                        # şema doğrulama
docker
```

## Hazır modeller / repolar

| Amaç | Kaynak | Not |
|---|---|---|
| Synthetic detection | `grip-unina/ClipBased-SyntheticImageDetection` | En pratik pretrained; zero-shot dene |
| Synthetic detection | Ojha vd. UniversalFakeDetect repo'su | Referans implementasyon |
| Localization | `SunnyHaze/IML-ViT` | Ağırlıklar + Colab demo mevcut |
| Localization | TruFor (GRIP-UNINA) | Noiseprint++ tabanlı, güçlü |
| IMDL çatısı | **IMDL-BenCo** | MantraNet, MVSS-Net, CAT-Net, PSCC-Net, TruFor, IML-ViT'i tek framework'te toplar. **Kendi eğitim döngünü yazma, bunu kullan.** |
| Segmentasyon | `facebook/sam-vit-base` | Maske üretimi |
| Inpainting | `runwayml/stable-diffusion-inpainting`, `diffusers` SDXL-inpaint | |
| Nesne silme | LaMa (`advimman/lama`) | AI olmayan temiz silme |
| Feature | `openai/clip-vit-large-patch14`, DINOv2 | |

> **Lisans uyarısı:** Her modelin ve veri setinin lisansını `docs/licenses.md`'ye yaz. Kurumsal bir ortamda "bu modeli ticari kullanabilir miyiz?" sorusu kesinlikle gelecek. Cevabı hazır olan stajyer, cevabı olmayandan farklıdır.

## Eğitim ortamı
- **Google Colab Pro** (T4/L4) veya **Kaggle Notebooks** (haftalık 30 saat ücretsiz P100/T4) — SDXL üretimi ve fine-tune için yeterli
- Üretimi **gece batch** olarak çalıştır, checkpoint'le, Drive'a yaz
- CLIP embedding'lerini **bir kez** çıkar, `.npy` cache'le → sonrası CPU'da laptop'ta döner
- Şirket GPU'su varsa erişim talebini **Hafta 1'de** aç (bürokrasi sürer)

---

# 6. Deney Planı (Konsolide)

Her deney için `experiments/EXX_isim/` klasörü: `config.yaml`, `run.py`, `results.json`, `plots/`, `NOTES.md`.

| ID | Hafta | Model / Yöntem | Eğitim verisi | Eğitim yöntemi | Metrik | Beklenen sonuç / Öğrenilecek |
|---|---|---|---|---|---|---|
| **E0** | 1 | ResNet-50 | 600 örnek | Fine-tune, 5 epoch | AUC | Pipeline çalışıyor mu (sonuç önemsiz) |
| **E1** | 3 | ResNet-50 | R+S full | Full fine-tune | AUC, TPR@1%FPR | Yüksek clean, düşük laundered |
| **E2** | 3 | ConvNeXt-T / EffNet-B0 | R+S full | Fine-tune | Aynı | Mimari farkı marjinal |
| **E3** | 3 | **CLIP ViT-L + LogReg** | R+S full | Frozen backbone + linear probe | Aynı | **Ana baseline.** Genellemede E1'i geçmeli |
| **E4** | 3 | ClipBased-SID (pretrained) | Eğitim yok | Zero-shot | Aynı | Literatür SOTA'sı bizim domain'de? |
| **E5** | 3 | E3 + laundering augment | R+S + augment | Linear probe | Laundered AUC | Robustness kazancı ölçülür |
| **E6** | 3 | E3, generator-disjoint | SD1.5+SDXL train / FLUX test | Linear probe | Unseen-gen AUC | Open-set genelleme sınırı |
| **E7** | 4 | **TruFor** (pretrained) | Yok | Zero-shot | Pixel-F1, IoU, det-AUC | Ana localization baseline |
| **E8** | 4 | **IML-ViT** (pretrained) | Yok | Zero-shot | Aynı | İkinci görüş |
| **E9** | 4 | MVSS-Net / CAT-Net | Yok | Zero-shot | Aynı | Üçüncü görüş |
| **E10** | 4 | En iyi IMDL + domain fine-tune | ~1.600 maskeli | Decoder-only, düşük LR | Aynı | Domain adaptasyon kazancı |
| **E11** | 4 | Maske→skor agregasyonu | — | — | det-AUC | max vs. mean vs. top-5% |
| **E12** | 5 | **Füzyon (LR + LGBM)** | Tüm sinyaller | 5-fold CV + kalibrasyon | AUC, TPR@1%FPR, ECE | Tekil modellerden üstün olmalı |
| **E13** | 5 | Ablation | — | Sinyal çıkararak | ΔAUC | Hangi sinyal ne kadar değerli |
| **E14** | 5 | Robustness matrisi | Test seti | — | 5 profil × 6 senaryo | Kırılganlık haritası |
| **E15** | 5 | Adversarial sanity | Test seti | — | Skor kayması | Sistem ne kadar kolay atlatılır |
| **E16** | 5 | Operasyonel simülasyon | Sentetik akış | — | İş metrikleri | Yönetici slaytı |
| **E17*** | 6 | VLM semantik tutarlılık | 20 örnek | Prompt | Kalitatif | Gelecek çalışma sinyali |

\* opsiyonel

**Deney disiplini kuralları:**
1. Her deneyde **tek bir şey** değiştir. İki şey değiştirirsen hangisinin etki ettiğini bilemezsin.
2. Random seed'i sabitle ve logla. **En az 3 seed** ile çalıştır, ortalama ± std raporla. Tek koşuluk sonuç bilim değildir.
3. Her deney bitiminde `NOTES.md`'ye **bir cümle hipotez sonucu** yaz: "Hipotez: X. Sonuç: doğrulandı/reddedildi çünkü Y."
4. Negatif sonuçları sil**me**. "ELA hiçbir katkı sağlamadı" bir bulgudur, sunumda anlatılır.

---

# 7. MVP Sistem Tasarımı

## 7.1. Uçtan uca akış

```
1. INPUT
   Görüntü yükleme (JPEG/PNG/HEIC), max 20MB, EXIF korunarak
   ↓
2. PREPROCESSING  (src/pipeline/preprocess.py)
   • Format doğrulama, HEIC→JPEG dönüşümü (EXIF saklanarak)
   • EXIF/metadata ham çıkarımı  ← ORİJİNAL DOSYADAN, resize'dan ÖNCE
   • JPEG kuantizasyon tablosu okuma
   • Görüntü hash (dedup / tekrar gönderim tespiti)
   • Model girdileri için resize kopyaları (CLIP 224, IMDL full-res)
   ↓
3. L1 METADATA ANALYSIS         → p_meta, flags[]
4. L2 SYNTHETIC DETECTION       → p_synth  (CLIP probe)
5. L3 MANIPULATION LOCALIZATION → p_manip, mask, mask_stats
6. L4 CLASSIC FORENSICS         → ela_energy, fft_score, dq_flag
   (3–6 paralel çalıştırılabilir — asyncio veya thread pool)
   ↓
7. FUSION  (src/fusion/model.py)
   feature_vector → calibrated_model → risk_score ∈ [0,1]
   ↓
8. DECISION & EXPLANATION
   risk_score < 0.30           → DÜŞÜK   → otomatik akışa devam
   0.30 ≤ risk_score < 0.70    → ORTA    → eksper incelemesi
   risk_score ≥ 0.70           → YÜKSEK  → derin inceleme + kanıt paketi
   + evidence[] listesi + overlay görsel
```

> **Kritik mühendislik detayı:** Metadata'yı **her zaman orijinal yüklenen dosyadan** çıkar. Eğer önce resize edip sonra EXIF okursan tüm metadata katmanın çöp olur. Bu, stajyer projelerinde en sık görülen sessiz hatadır.

## 7.2. API sözleşmesi

```json
POST /analyze   (multipart/form-data: file)

200 OK
{
  "request_id": "a3f21c...",
  "image_hash": "sha256:...",
  "risk_score": 0.83,
  "risk_band": "HIGH",
  "recommended_action": "MANUAL_REVIEW",
  "signals": {
    "synthetic_probability": 0.14,
    "manipulation_probability": 0.79,
    "manipulated_area_ratio": 0.042,
    "metadata_score": 0.65
  },
  "evidence": [
    {"type": "LOCAL_MANIPULATION", "severity": "high",
     "message": "Sol ön kapı bölgesinde görüntünün %4.2'sini kaplayan manipülasyon izi",
     "region": {"x": 412, "y": 233, "w": 180, "h": 140}},
    {"type": "MISSING_EXIF", "severity": "medium",
     "message": "EXIF verisi bulunamadı; görüntü yeniden kaydedilmiş olabilir"},
    {"type": "DOUBLE_JPEG", "severity": "medium",
     "message": "Çift JPEG sıkıştırma izi tespit edildi"}
  ],
  "artifacts": {
    "heatmap_url": "/explain/a3f21c.../heatmap.png",
    "overlay_url": "/explain/a3f21c.../overlay.png"
  },
  "model_version": "forensics-v0.3.1",
  "processing_time_ms": 1180
}
```

**Neden bu şema iyi:** Skoru, sinyalleri ve insan-okunur kanıtı ayırıyor. Eksper `evidence[]`'ı okur, veri bilimci `signals`'a bakar, sistem `risk_band`'e göre yönlendirir. `model_version` alanı üretim izlenebilirliği için şart.

## 7.3. Demo UI (Gradio)

Üç sekme yeter:
1. **Analiz** — sürükle-bırak → skor kartı (renkli bant) + kanıt listesi + overlay
2. **Karşılaştırma** — orijinal | ısı haritası | ELA | frekans spektrumu (4'lü grid, forensic hissi verir)
3. **Batch** — klasör yükle → tablo + sıralama (yöneticiye "ölçek" hissi verir)

**Demo taktiği:** Yönetici sunumunda **canlı yükleme yapma riski alma.** Önceden 6 vaka hazırla (2 gerçek, 2 inpaint, 1 tam sentetik, 1 zor/başarısız) ve sistemin nasıl davrandığını ezberle. Ama **canlı yükleme seçeneğini de hazır tut** — biri kendi telefonundan fotoğraf atmak isterse bu unutulmaz bir an olur. Riski azaltmak için: yerelde çalıştır, internet bağımlılığı olmasın.

## 7.4. Performans hedefleri
- Tek görüntü uçtan uca: **< 3 saniye** (GPU), < 10 sn (CPU)
- Bellek: < 8 GB VRAM
- Batch throughput: > 20 görüntü/dk

---

# 8. Başarı Kriterleri ve Metrikler

## 8.1. Metrikler ve sigorta anlamları

| Metrik | Tanım | Sigortada ne demek |
|---|---|---|
| **Accuracy** | Doğru tahmin oranı | ⚠️ **Neredeyse anlamsız.** Gerçek dağılımda sahteler %1 ise, "hepsi gerçek" diyen model %99 accuracy alır. Sunumda tek başına gösterme. |
| **Precision** | İşaretlediklerimin kaçı gerçekten sahte | **Eksper zamanının verimliliği.** Düşükse eksper 10 dosya inceler 1'i sahte çıkar, sisteme güvenmeyi bırakır. |
| **Recall (TPR)** | Sahtelerin kaçını yakaladım | **Kaçan dolandırıcılık = doğrudan ödenen para.** |
| **F1** | Dengeli özet | Rapor için iyi, karar için yetersiz |
| **ROC-AUC** | Eşikten bağımsız ayırt etme gücü | Model karşılaştırması için ana metrik |
| **PR-AUC** | Dengesiz veride ROC'tan bilgilendirici | Sahte oranı düşük olduğu için **ROC-AUC'tan daha dürüst** |
| **TPR @ FPR=1%** | %1 yanlış alarm bütçesinde yakalama oranı | ⭐ **Operasyonel ana metrik.** "Günde 1000 dosyadan 10'unu boşuna işaretlemeye razıyız, karşılığında sahtelerin %X'ini yakalıyoruz." |
| **Confusion Matrix** | 4 hücre | Sunumda mutlak sayılarla göster, yüzdeyle değil |
| **ECE / kalibrasyon** | 0.8 skoru gerçekten %80 mi | Risk bantları ancak kalibre skorla anlamlıdır |
| **Pixel F1 / IoU** | Maske doğruluğu | Ekspere gösterilen bölge gerçekten doğru mu |
| **FP area rate** | Temiz görüntülerde işaretlenen alan | Gürültülü ısı haritası ekspere güven kaybettirir |

## 8.2. Yanlış negatif mi, yanlış pozitif mi? — Nüanslı cevap

Sen soruda "yanlış negatiflerin neden kritik olduğunu" anlatmamı istedin. Doğru, ama **eksik ve tek başına söylersen yönetici sana itiraz eder.** Gerçek cevap şudur:

**Yanlış negatif (kaçan sahte):**
- Doğrudan finansal kayıp (ödenen sahte hasar)
- Sistemik risk: bir kez işe yarayan yöntem forumlarda yayılır, bir vaka yüzlerce olur
- Tespit edilemez: kaçan sahteyi hiç öğrenemezsin, yani metriğin bile yoktur

**Yanlış pozitif (masum müşteriyi işaretleme):**
- Müşteri memnuniyeti ve NPS kaybı, süreç gecikmesi
- Şikayet, Sigorta Tahkim, itibar riski
- **Eksper güven kaybı** — en sinsi olanı: yanlış alarm oranı yüksek bir sistem 3 ay sonra kimse tarafından kullanılmaz, teknik olarak "başarılı" projeler böyle ölür
- Regülasyon riski: otomatik kararın müşteri aleyhine sonuç doğurması

**Doğru çerçeve — bunu sunumda söyle:**
> "Sistem sahtekârlık kararı vermiyor, bir inceleme kuyruğu önceliklendiriyor. Bu yüzden ana metriğimiz Accuracy değil, **sabit bir yanlış alarm bütçesinde yakalama oranı (TPR @ FPR = %1)**. Yanlış alarm bütçesini operasyon ekibinin eksper kapasitesi belirler, biz modeli o kısıta göre kalibre ederiz."

Bu cümle seni stajyerden Ar-Ge mühendisine terfi ettirir. Çünkü modeli iş kısıtına bağlıyorsun.

## 8.3. Gerçekçi hedefler (bunları taahhüt et)

| Hedef | Minimum | İyi | Mükemmel |
|---|---|---|---|
| Tam sentetik tespiti, `clean` | AUC 0.90 | 0.95 | 0.98 |
| Tam sentetik tespiti, `whatsapp` | AUC 0.75 | 0.85 | 0.92 |
| Tam sentetik, **görülmemiş generator** | AUC 0.70 | 0.82 | 0.90 |
| Lokal inpaint tespiti (görüntü sev.), `clean` | AUC 0.70 | 0.82 | 0.90 |
| Lokal inpaint tespiti, `whatsapp` | AUC 0.60 | 0.72 | 0.82 |
| Localization pixel-F1 (tespit edilenlerde) | 0.35 | 0.50 | 0.65 |
| Uçtan uca gecikme | < 5 sn | < 3 sn | < 1.5 sn |

> **Bu tablo senin dürüstlük sigortandır.** Sunumdan önce beklentiyi kalibre eder. Localization F1'in 0.5 civarında çıkması normaldir ve literatürle uyumludur; yönetici 0.95 bekliyorsa ve sen 0.5 alırsan proje "başarısız" görünür. Hedefleri **önceden** paylaş.

## 8.4. Raporlama formatı

Her deney sonucu şu tablo formatında:

```
Deney: E3 — CLIP ViT-L/14 + Linear Probe
Veri: test_v1 (donmuş), n=1440
Seed: 3 koşu ortalaması ± std

Senaryo         | clean          | whatsapp       | aggressive
----------------|----------------|----------------|---------------
Full synthetic  | .962 ± .004    | .881 ± .009    | .804 ± .014
Inpaint-add     | .714 ± .011    | .663 ± .015    | .601 ± .021
Inpaint-remove  | .688 ± .013    | .642 ± .018    | .587 ± .019
Classic manip   | .751 ± .008    | .702 ± .012    | .655 ± .017
(değerler ROC-AUC)

Bulgu: Model tam sentetik üretimde güçlü, bölgesel düzenlemede zayıf.
       Bu, L3 localization katmanının gerekliliğini doğruluyor.
```

(Rakamlar örnektir — kendi ölçümlerinle doldur.)

---

# 9. Staj Sonu Sunum Planı

**Format:** 18–22 dakika sunum + 3 dk demo + 5–10 dk soru. 15–18 slayt.
**Kitle:** Karma — teknik olmayan yöneticiler + IT/veri ekibi. **Teknik olmayana göre yaz, teknik detayı yedek slaytlara koy.**

| # | Slayt | İçerik | Görsel |
|---|---|---|---|
| 1 | Kapak | Proje adı, sen, tarih, "Türkiye Sigorta AI Ar-Ge Stajı" | Temiz |
| 2 | **Problem** | Dijital hasar ihbarında görsel kanıtın güvenilirliği; Verisk 2026: sigortacıların %98'i AI editleme araçlarının dijital dolandırıcılığı büyüttüğünü söylüyor, sadece %32'si tespit konusunda kendine güveniyor | Büyük tek istatistik |
| 3 | **Tehdit anatomisi** | 6 saldırı senaryosu, olasılık × zorluk matrisi | 2×3 ikon grid |
| 4 | **"Bunu ayırt edebilir misiniz?"** | 4 fotoğraf: 2 gerçek 2 manipüle. **Cevabı 30 saniye bekletip sor.** | Salonu uyandırır — en etkili slayt |
| 5 | Kapsam ve varsayımlar | Ne yaptım, ne yapmadım, hangi veriyi kullandım (gerçek müşteri verisi YOK) | Metin |
| 6 | **Yaklaşım** | 3 katmanlı mimari şeması, "karar değil triage" ilkesi | Mimari diyagramı |
| 7 | Literatür özeti | 4 ana yaklaşım, neden CLIP + IMDL seçtim | Karşılaştırma tablosu |
| 8 | **Veri stratejisi** | Sentetik veri üretim hattı, 4.800 görüntü, 6 senaryo | Üretim akış şeması + örnek kolaj |
| 9 | Deney tasarımı | 17 deney, generator-disjoint split, laundering profilleri | Deney matrisi |
| 10 | **Sonuçlar 1: Tespit** | ROC eğrileri, model karşılaştırma tablosu | ROC grafiği |
| 11 | **Sonuçlar 2: Robustness** | Sıkıştırma profil × senaryo ısı haritası. "Laboratuvar %96 → WhatsApp %88" | Heatmap ⭐ |
| 12 | **Sonuçlar 3: Lokalizasyon** | Kalitatif galeri: 3 başarı, 2 başarısızlık | Görsel grid ⭐⭐ |
| 13 | **Ablation** | Hangi sinyal ne kadar katkı veriyor | Yatay bar grafik |
| 14 | **Operasyonel etki** | E16 simülasyonu: "10.000 dosya, eşik 0.8 → 120 inceleme, sahtelerin %78'i" | İş metriği grafiği ⭐⭐⭐ |
| 15 | **DEMO** | Canlı 3 dakika | — |
| 16 | **Limitasyonlar** | Dürüst liste: sentetik veri gerçek dolandırıcıyı temsil etmeyebilir, ağır sıkıştırmada performans düşüyor, adversarial dayanıklılık test edilmedi | Metin — **bu slayt sana saygı kazandırır** |
| 17 | **Yol haritası** | Faz 1 (3 ay): gerçek dosyalarla gölge mod. Faz 2 (6 ay): mobil app'te C2PA imzalı çekim. Faz 3: eksper geri bildirim döngüsü | Zaman çizelgesi |
| 18 | Kapanış | Ne öğrendim, kod/dokümantasyon nerede, teşekkür | Repo linki |

**Yedek slaytlar (Q&A için):** Model mimarisi detayı, hiperparametreler, kalibrasyon eğrisi, tam sonuç tabloları, lisans/KVKK notları, maliyet tahmini.

## Sunum taktikleri

1. **Slayt 4 ile başla, teknikle değil.** İnsanlara kendi gözlerinin yetmediğini gösterirsen sonrasında her şeyi dinlerler.
2. **Bir sayı seç ve tekrarla.** Örneğin "sahtelerin %78'i, incelemenin %1.2'si ile". Yöneticiler tek sayı hatırlar.
3. **AUC'u yönetici slaytlarında kullanma.** "Model 100 dosyadan 78'ini doğru önceliklendiriyor" de.
4. **Limitasyon slaytını atlama.** Deneyimli yöneticiler zayıflığını söylemeyene güvenmez. Sen söylersen, soru sormazlar.
5. **"Bu yarın devreye alınabilir mi?" sorusuna hazır ol.** Doğru cevap: *"Hayır. Önce 3 ay gölge modda gerçek dosyalar üzerinde çalışması, eksper geri bildirimiyle kalibre edilmesi gerekir. Bu prototip o çalışmanın altyapısını hazır ediyor."* Bu cevap seni ciddiye aldırır.
6. **Demoda başarısızlık örneği de göster.** Sadece başarı gösteren demo, satış sunumudur; ikisini gösteren demo, mühendislik sunumudur.

---

# 10. Kritik Tavsiyeler — Zaman Katilleri ve Tuzaklar

## 10.1. Yapma (proje katilleri)

**1. Sıfırdan model mimarisi tasarlama.**
6 haftada olmaz. Literatürdeki en güçlü baseline'ı al, domain'e uyarla, dürüstçe ölç. Bu daha değerlidir.

**2. Veri üretimini "sonra hallederim" deme.**
En sık ölüm sebebi. Hafta 2'yi tamamen veriye ayır. Kötü veriyle iyi model olmaz; iyi veriyle vasat model bile iş görür.

**3. Yüksek AUC'a sevinme, önce sızıntı ara.**
AUC 0.99 gördüğünde ilk tepkin şüphe olmalı. Kontrol listesi: real/fake çözünürlük eşit mi, format eşit mi, JPEG kalitesi eşit mi, prompt estetiği kestirme yol açıyor mu, source-image sızıntısı var mı. **32×32 testi** yap.

**4. Aynı generator'da test etme.**
SD1.5'te eğitip SD1.5'te test etmek anlamsızdır. Test setinde mutlaka görülmemiş bir generator tut.

**5. ELA'yı ana yöntem yapma.**
Görsel olarak ikna edici olması bilimsel olarak geçerli olduğu anlamına gelmez. Modern diffusion inpainting'de neredeyse hiçbir şey göstermez. Feature olarak ekle, ablation'da katkısını göster, orada bırak.

**6. Sadece `clean` görüntülerde ölçme.**
Sigorta gerçeği WhatsApp'tır. Laundering testi yapmayan bir rapor, üretimde çöker.

**7. Localization için sıfırdan eğitim denemesi.**
Elinde ~1.600 maskeli örnek var. TruFor/IML-ViT ImageNet + büyük forensics korpuslarıyla eğitildi. Sen zero-shot kullan, gerekirse decoder fine-tune et. Sıfırdan eğitim 3 haftanı yer ve sonuç daha kötü olur.

**8. Son haftaya kadar sunum ve dokümantasyona başlamama.**
Haftalık raporlarını düzenli yazarsan Hafta 6'da sunum 1 günde çıkar. Yazmazsan son hafta panikte kod da bitmez, sunum da.

**9. Metadata katmanını küçümseme.**
"Basit, akademik değil" diye atlama. Maliyeti 1 gün, katkısı yüksek ve **yöneticinin anladığı tek katman** o.

**10. Metadata'yı resize edilmiş görüntüden okuma.** (Bkz. 7.1) Sessiz ve öldürücü hata.

**11. Mükemmeliyetçilik.**
Hafta 5'te "modeli biraz daha iyileştireyim" tuzağına düşme. **Çalışan bir sistem + dürüst ölçüm > mükemmel model + yarım demo.** AUC'u 0.86'dan 0.88'e çıkarmak kimsenin umurunda değil; çalışan bir demo herkesin umurunda.

**12. Gerçek müşteri verisini izinsiz kullanma.** KVKK. Erişimin olsa bile önce hukuk/uyum onayı sor. "Sormadım ama kullandım" bir stajı bitirir.

## 10.2. Yap (kazandıranlar)

1. **Hafta 1'de uçtan uca dikey dilim çıkar.** Kötü model + kötü veri ama çalışan zincir. Sonrası iyileştirmedir.
2. **Her şeyi manifest'e yaz.** Klasör yapısına asla güvenme.
3. **CLIP embedding'lerini cache'le.** Deney hızın 100× artar.
4. **Deneyleri MLflow/W&B'ye kaydet.** Hafta 6'da "hangi koşuydu bu?" sorusunu sormamak paha biçilmez.
5. **Kendi telefonunla 100 fotoğraf çek.** Tam EXIF'li gerçek veri, sentetikle karşılaştırman için altın standart.
6. **Haftalık 1 sayfa rapor yaz.** Hem mentörün seni takip eder hem dokümantasyonun kendiliğinden oluşur.
7. **Negatif sonuçları raporla.** "ELA katkı sağlamadı", "generator-disjoint testte performans %12 düştü" — bunlar bulgudur.
8. **3 seed ile çalıştır, ± std raporla.**
9. **Test setini dondur ve dokunma.**
10. **Kalitatif galeri biriktir.** Her hafta ilginç 5 örnek kaydet. Sunumun en ikna edici materyali bu olacak.
11. **Bir "insan baseline"ı ölç.** 20 kişiye 20 görüntü göster. "İnsan %62, sistem %88" cümlesi çok güçlüdür.
12. **Model card ve limitations dokümanı yaz.** Regüle sektörde bu belgeler teknik rapordan çok okunur.

## 10.3. Öncelik sıralaması (zaman daralırsa neyi keseceksin)

**Asla kesme (P0):**
- Veri üretim hattı + laundering + doğru split
- CLIP tabanlı synthetic detection (E3/E4)
- Bir IMDL modeli zero-shot (E7 veya E8)
- Ölçüm altyapısı ve dürüst raporlama
- Çalışan bir demo (Gradio yeter)

**Zaman varsa (P1):**
- Füzyon + kalibrasyon
- Ablation ve robustness matrisi
- Operasyonel simülasyon (E16)
- FastAPI + Docker

**Lüks (P2 — kesilebilir):**
- IMDL fine-tune (E10)
- Klasik forensics feature'ları (ELA/FFT)
- VLM semantik kontrolü
- Batch UI

---

# 11. Nihai Çıktılar

## 11.1. GitHub reposu

**`insurance-image-forensics`** — Türkçe/İngilizce README, MIT veya kurumsal lisans.

```
insurance-image-forensics/
├── README.md                    # problem, mimari, kurulum, sonuç özeti, demo GIF
├── docs/
│   ├── technical_report.md      # 10-15 sayfa ana rapor
│   ├── dataset_card.md
│   ├── model_card.md
│   ├── limitations.md
│   ├── licenses.md
│   ├── lit/                     # makale notları
│   └── weekly/W1..W6.md
├── src/
│   ├── data/{generators,launder,manifest,splits}.py
│   ├── detectors/{base,metadata,clip_probe,cnn_baseline,
│   │              trufor_wrapper,imlvit_wrapper,classic}.py
│   ├── features/clip_embed.py
│   ├── fusion/{feature_builder,model,calibrate}.py
│   ├── explain/{gradcam,overlay,evidence}.py
│   ├── eval/{metrics,localization_metrics,report}.py
│   ├── pipeline/analyze.py      # uçtan uca orkestrasyon
│   └── api/main.py              # FastAPI
├── app/gradio_app.py
├── configs/                     # YAML deney konfigleri
├── experiments/E00..E17/
├── notebooks/                   # keşif, final grafikler
├── tests/
├── Dockerfile, docker-compose.yml
└── requirements.txt
```

**README'de olması gerekenler (işe alım açısından kritik):**
- Bir demo GIF'i (yükleme → skor → ısı haritası)
- Mimari diyagramı
- Ana sonuç tablosu
- "Bilinen limitasyonlar" bölümü
- `docker compose up` ile 1 komutta çalışan kurulum

## 11.2. Teslim edilebilir çıktı listesi

| # | Çıktı | Kime hitap eder |
|---|---|---|
| 1 | Çalışan MVP (Gradio + FastAPI + Docker) | Yönetici, IT |
| 2 | Sentetik veri seti (~4.800, maskeli, manifest'li) + üretim kodu | Veri ekibi — **şirkette kalan kalıcı varlık** |
| 3 | 17 deneyin sonuç tabloları ve grafikleri | Veri bilimci |
| 4 | Teknik rapor (10–15 sayfa) | Ar-Ge |
| 5 | Model card + limitations + licenses | Uyum/hukuk |
| 6 | Yönetici sunumu (18 slayt) | Yönetim |
| 7 | Benchmark harness (yeni model eklendiğinde tek komutla ölçüm) | **En değerli mühendislik çıktısı** |
| 8 | Gelecek yol haritası (gölge mod → C2PA → geri bildirim döngüsü) | Ürün |

## 11.3. Bunun ötesine geçmek istersen (staj sonrası)

- **Bitirme projesi / makale:** "Domain-Specific Evaluation of Synthetic Image Detectors on Vehicle Damage Photographs under Realistic Compression" — bu, gerçek bir literatür boşluğu. Yerel bir konferans veya arXiv preprint'i erişilebilir bir hedef.
- **Açık veri katkısı:** Sentetik araç hasarı forensics veri setini (lisans uygunsa) Hugging Face'te yayınlamak. Bu tür domain-spesifik set literatürde yok.
- **Faz 2 önerisi:** Şirket mobil uygulamasında in-app çekim + C2PA imzalama. Problemi tespit etmekten **önleme** seviyesine taşır ve çok daha yüksek ROI'lidir. Sunumunun son slaytında bunu öner.

---

# 12. Son Söz

Bu projenin başarısı modelin AUC'u ile ölçülmeyecek. Şu üç şeyle ölçülecek:

1. **Çalışan bir şey teslim ettin mi?** Vasat ama çalışan > mükemmel ama yarım.
2. **Ölçümlerin dürüst mü?** Sızıntısız split, laundering testi, görülmemiş generator, negatif sonuçların raporlanması. Bir kıdemli mühendis sunumunda ilk bunlara bakar.
3. **Sınırlarını biliyor musun?** "Bu sistem şunu yapamaz" diyebilen bir mühendis, "her şeyi yapar" diyenden çok daha güvenilirdir.

Sen NLP'den geliyorsun. CV'ye geçişte iyi haber şu: pipeline disiplini, deney tasarımı, metrik okuryazarlığı ve sızıntı avcılığı **transfer edilebilir** ve bunlar işin %70'i. Öğrenmen gereken kalan %30 — konvolüsyon, segmentasyon, görüntü formatları, forensic izler — 2 haftada kapanır.

**Bu hafta yapman gereken tek şey:** repo'yu aç, `Detector` arayüzünü yaz, CarDD'yi indir, 20 sentetik görüntü üret ve E0'ı çalıştır. Cuma günü kötü ama çalışan bir zincirin olsun. Gerisi gelir.

Başarılar.
