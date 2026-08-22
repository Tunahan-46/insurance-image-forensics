# Dataset Card — Insurance Image Forensics v2

**Sürüm:** v2 (2026-08-20)
**Manifest:** `data/processed/manifest_v2.parquet` (5.222 satır)
**Değerlendirme manifesti:** `data/processed/manifest_v2_laundered.parquet` (17.176 satır)
**Üreten kod:** `scripts/build_manifest_v2.py`, `scripts/apply_laundering.py`
**Lisans durumu:** aşağıya bakınız — **veri seti bu haliyle yeniden dağıtılamaz**

---

## 1. Ne için var

Sigorta hasar dosyalarına yüklenen araç fotoğraflarında iki ayrı problemi ölçmek için:

- **Görev A — tam sentetik tespiti:** fotoğrafın tamamı bir difüzyon modeliyle
  üretilmiş mi? (`real` vs `fully_synthetic`)
- **Görev B — manipülasyon tespiti + konumlandırma:** gerçek bir fotoğraf üzerinde
  bölgesel değişiklik yapılmış mı, yapıldıysa nerede? (`real` vs `manipulated`,
  + piksel maskesi)

Sistem **karar verici değil triage** amaçlıdır (plan bölüm 2): şüpheli dosyayı
insan eksperine yönlendirir, tek başına ret/kabul üretmez.

---

## 2. Katmanlar ve sayılar

| Kod | Katman | Adet | Kaynak | Maske |
|---|---|---|---|---|
| R | Gerçek — CarDD | 4.000 | CarDD_COCO | — |
| R | Gerçek — kendi telefon fotoğraflarım | 52 | iPhone 13, tam EXIF | — |
| S | Tam sentetik | 330 | SD1.5 / SDXL / SD-Turbo | — |
| M1 | Inpaint ile hasar **ekleme** | 108 | CarDD + SDXL-inpaint | ✅ |
| M1b | Inpaint ile hasar **büyütme** | 92 | CarDD + SDXL-inpaint | ✅ |
| M2 | Inpaint ile hasar **silme** | 120 | CarDD + SDXL-inpaint / OpenCV | ✅ |
| M3 | Klasik manipülasyon | 520 | CarDD (copy-move / splice / bg_replace) | ✅ |
| | **Toplam** | **5.222** | | 840 maske |

Manipüle edilmiş 840 görüntünün **tamamında** piksel maskesi var (840/840).

### Alt kırılım

**S — tam sentetik (330):**

| Generator | train | val | test | Rol |
|---|---|---|---|---|
| `sd15` | 97 | 26 | 27 | ana üretici |
| `sdxl` | 59 | 12 | 9 | ana üretici |
| `sd_turbo` | 0 | 0 | **100** | **test-only** (görülmemiş generator) |

**M — manipülasyon (840):**

| manip_type | Adet | Yöntem |
|---|---|---|
| `splice` | 210 | başka fotoğraftan bölge yapıştırma (donör aynı split'ten) |
| `copy_move` | 208 | aynı fotoğraf içinde bölge kopyalama |
| `inpaint_remove` | 120 | 60× SDXL-inpaint, 60× OpenCV Telea |
| `inpaint_add` | 108 | SDXL-inpaint, mevcut hasarın DIŞINA yeni hasar |
| `bg_replace` | 102 | arka plan değiştirme |
| `inpaint_enlarge` | 92 | SDXL-inpaint, mevcut hasarı halka maskesiyle büyütme |

---

## 3. Nasıl üretildi

### S — tam sentetik

Kombinatoryal prompt motoru (`src/data/generators/prompts.py`, ~28.5M kombinasyon).
Elle prompt yazılmaz. Her prompt zorunlu olarak bir "amatör kalite" ve bir "kamera"
özelliği taşır — plan 4.5 Tuzak 4 (estetik sızıntısı) için.

Örnek pozitif prompt:

> `quick documentation photo of a silver minibus, flat tire, on a city street, overcast daylight, wide shot, shot on iPhone`

Ortak negatif prompt (kısaltılmış):

> `professional, cinematic, 8k, ultra detailed, artstation, illustration, render, 3d, cartoon, anime, painting, oversaturated, studio lighting, perfect composition, watermark, text, logo, hdr, bokeh, ...`

Her üretim `gen_log.jsonl`'e loglanır: model, repo, seed, steps, guidance,
çözünürlük, prompt bileşenleri (araç/renk/hasar/panel/ortam).

### M1 / M2 — inpaint tabanlı

Kaynak: CarDD gerçek fotoğrafı + **insan anotasyonlu hasar maskesi** (W1 Bulgu 2 —
4000/4000 eşleşme, SAM ile maske üretmeye gerek kalmadı).

- **M1 (add):** `mask_for_damage_add` mevcut hasarın **dışında** bir bölge seçer
- **M1b (enlarge):** `mask_for_damage_enlarge` halka maskesi (yeni \ eski)
- **M2 (remove):** `mask_for_removal` hasar maskesi + pay; maske alanı >%35 ise reddedilir
  (model "yeni bir araba" uydurur, senaryo gerçekçilikten çıkar)

Maskeler düzensiz ve yumuşak kenarlıdır — dikdörtgen maske dedektöre kestirme yol verir.

**Kabul kapısı:** üretilen görüntü maske içinde yeterince piksel değiştirmediyse
diske **yazılmaz** (`MIN_CHANGED_IN_MASK`). Maske dışına sızıntı da ölçülüp loglanır.

### M3 — klasik manipülasyon

GPU'suz. `copy_move`, `splice`, `bg_replace`. Donör görüntü **her zaman aynı
split'ten** seçilir (`pick_donor`), aksi halde split sızıntısı olurdu.

---

## 4. Laundering (dağıtım kanalı simülasyonu)

Gerçek dünyada fotoğraf sigortacıya ham gelmiyor: WhatsApp'tan geçiyor, ekran
görüntüsü alınıyor, yeniden kaydediliyor. Her kaynak görüntünün laundered
kopyaları üretilir ve **eğitim/değerlendirme yalnızca bu kopyalar üzerinden yapılır.**

| Profil | İşlem | JPEG kalite |
|---|---|---|
| `clean` | yok, sadece yeniden kaydet | q95 |
| `whatsapp` | uzun kenar → 1600 | q75 |
| `screenshot` | 1280 + kırpma + PNG ara adım | q90 |
| `double_jpeg` | q95 → yeniden sıkıştır | q70 |
| `aggressive` | 1024 + blur σ0.5 | q60 |

**Train/val'de 3 profil** (`clean`, `whatsapp`, `double_jpeg`),
**test'te 5 profil** — test seti dağıtım kanalına karşı stres testine tabi tutulur.
Maskeler laundering geometrisiyle senkron dönüştürülür ama **asla JPEG'lenmez**
(PNG kalır; JPEG artefaktı ikili maskeyi bozar).

Sonuç: 5.222 kaynak → **17.176 değerlendirme örneği**.

| split | profil başına | toplam |
|---|---|---|
| train | 3.472 × 3 | 10.416 |
| val | 995 × 3 | 2.985 |
| test | 755 × 5 | 3.775 |

---

## 5. Split kuralları

**Kaynak split dağılımı sabittir: 2854 / 817 / 381** (train/val/test).
CarDD'nin kendi küratörlü bölümünden (`train2017`/`val2017`/`test2017`) gelir;
kendi fotoğraflarım için id hash'inden deterministik türetilir.

Üç kural makine tarafından zorlanır (`build_manifest_v2.py` her çalıştığında doğrular):

1. **Source-image-disjoint.** Bir CarDD fotoğrafı ve ondan türetilen **her şey**
   aynı split'te kalır (`source_image_id` gruplama anahtarı). Çatışma olursa
   türetilmiş görüntü elenir, kaynak taşınmaz. — plan 4.5 Tuzak 1
2. **Generator-disjoint.** `sd_turbo` ve `flux_schnell` test-only olarak
   işaretlidir; train/val'de görünürlerse manifest üretimi durur.
3. **image_id benzersizliği.** `image_id` ≠ `source_image_id`: türetilmiş satırlar
   kendi `variant_id`'sini taşır. (W2 Bulgu 8 — bu ayrım yapılmadan önce
   laundering, manipülasyon kopyasını gerçeğin üzerine yazıyordu.)

**Test seti donduruldu** (2026-08-20): `data/processed/test_manifest_frozen.parquet`,
755 satır.

```
sha256: d0e2cb375c3d9602a132dc068fe2fecd5f6850e4a5736c5b305245942ef6f6cc
```

Plan 4.5 Tuzak 5 gereği Hafta 5-6'ya kadar açılmayacak: test setine bakarak
model/hiperparametre seçmek kendini kandırmaktır. Model seçimi **val** üzerinden
yapılır.

---

## 6. Bilinen sorunlar ve kısıtlar

### ⚠️ Görev A'da çözünürlük/metadata kestirme yolu AÇIK

Ölçüldü ve raporlandı (`experiments/E1_shortcut/`, W3 Bulgu 9-10):

| prob | Görev A (real vs synth) | Görev B (real vs manip) |
|---|---|---|
| `meta` (piksel okumaz) | **0.974 – 0.988 ALARM** | 0.518 – 0.541 ok |
| `px8` | **0.776 – 0.778 ALARM** | 0.498 – 0.500 ok |
| `px32` | 0.644 – 0.660 | 0.476 – 0.484 ok |

Sebep, çözünürlük dağılımlarının örtüşmemesi:

| label | uzun kenar min | medyan | max |
|---|---|---|---|
| `fully_synthetic` | 512 | **768** | 1152 |
| `real` | 1000 | **1000** | 4032 |
| `manipulated` | 1000 | **1000** | 4032 |

Laundering format sızıntısını (Tuzak 3) kapatıyor ama **çözünürlük sızıntısını
(Tuzak 2) kapatmıyor** — beş profilde de meta AUC ~0.98'de sabit kalıyor.

**Sonuç: Görev A üzerinde alınan yüksek AUC değerleri, bu düzeltilmeden
forensic sinyal olarak yorumlanamaz.** Görev B için böyle bir kısıt yok.

### Diğer kısıtlar

- **Ölçek plan hedefinin altında.** Plan S:1200 / M1:800 / M2:400 öngörüyordu;
  elimizde 330 / 200 / 120 var (GPU bütçesi nedeniyle revize hedefler: 300/250/150).
- **FLUX hiç üretilmedi.** Test-only generator rolünü `sd_turbo` (100 görüntü)
  üstleniyor. "Görülmemiş generator" deneyi bu nedenle FLUX değil SD-Turbo üzerinden
  ölçülecek — SD-Turbo mimari olarak SD'ye yakın olduğu için bu **kolay** bir
  genelleme testidir; gerçek zorluk FLUX'ta olurdu.
- **`real` katmanı tek bir kaynağa yaslanıyor.** 4.000/4.052 CarDD'den. Kamera
  çeşitliliği, sahne çeşitliliği ve coğrafi kapsam CarDD'nin kapsamıyla sınırlı.
- **CarDD'de EXIF yok.** Akademik set, metadata temizlenmiş. L1 metadata dedektörü
  yalnızca 52 kendi fotoğrafım üzerinde anlamlı ölçülebilir (W1 Bulgu 4).
- **Sınıf dengesizliği:** gerçek 4.052 / sahte 1.170 (≈3.5:1).

---

## 7. Lisans, etik, KVKK

- **CarDD** kendi lisansı altındadır ve bu repoda **yeniden dağıtılmaz**;
  `data/` tamamen `.gitignore` içindedir. Kullanmak isteyen CarDD'yi kendi
  kaynağından indirip `scripts/build_manifest_v2.py` ile manifesti yeniden üretmelidir.
- **Türetilmiş görüntüler** (M1/M2/M3) CarDD'den türetildiği için aynı lisans
  kısıtlarına tabidir.
- **Kendi telefon fotoğraflarım** (52) yalnızca bu çalışma için çekildi.
- **Difüzyon model çıktıları** ilgili modellerin lisanslarına tabidir
  (SD1.5, SDXL-inpainting, SD-Turbo). SD1.5/SDXL çıktılarında safety checker
  kapatılmıştır — çıktılar halka açık bir serviste filtresiz sunulmamalıdır.
- **KVKK — açık risk:** plakalar **henüz bulanıklaştırılmadı**. Araştırma akışında
  risk yok (`data/` gitignore'da), ancak veri seti paylaşımı, sunum görseli veya
  bu kartın kolajı **öncesinde** anonimleştirilmelidir. Bu nedenle görsel kolajda
  kendi telefon fotoğraflarım kullanılmaz.
- **Gerçek müşteri/hasar dosyası verisi kullanılmamıştır** ve varsayılan plan
  hiç kullanmamaktır.

---

## 8. Yeniden üretim

```powershell
# 1. CarDD'yi data/raw/cardd/ altına yerleştir
# 2. Klasik manipülasyonlar (GPU'suz, ~12 dk)
python -m src.data.generators.classic_manip --n 520

# 3. Difüzyon katmanları (GPU gerekir — Kaggle T4x2, ~3 saat)
#    notebooks/W2_kaggle_uretim.ipynb

# 4. Manifest + laundering
python scripts\build_manifest_v2.py
python scripts\apply_laundering.py
python scripts\build_manifest_v2.py --freeze-test

# 5. Kestirme yol teşhisi (plan 4.5 — zorunlu)
python scripts\run_e1_shortcut.py
```

Tüm üretim seed'lidir ve `gen_log.jsonl` dosyalarında kayıtlıdır.
Üreticilerin `resume=True` varsayılanı vardır: mevcut dosyalar atlanır,
sıfırdan üretim için çıktı klasörünün silinmesi gerekir.
