# DEVİR TESLİM ÖZETİ — Insurance Image Forensics

**Tarih:** Hafta 2 ortası | **Repo:** github.com/Tunahan-46/insurance-image-forensics
**Son commit:** `f4b95b2` (push edildi, main → origin/main)
**Yerel yol:** `C:\Users\tunah\OneDrive\Masaüstü\insurance-image-forensics`
**Ortam:** Windows + PowerShell, `.venv`, Python 3.13, **OpenCV 5.0.0**

---

## 0. Bu projede önce okunması gereken dosya

`docs/mentorluk_plani.md` — 6 haftalık mentorluk planı ve teknik yol haritası.
Proje bu plana göre ilerliyor, plandaki bölüm numaraları (4.1, 4.5, 7.1...) kod
yorumlarında referans olarak geçiyor. **Bir karar vermeden önce plandaki ilgili
bölüme bak.**

`docs/weekly/W1.md` — Hafta 1 raporu ve dört bulgusu. Özellikle Bulgu 3
(E0'da AUC 0.364) ve Bulgu 4 (CarDD'de EXIF yok) hâlâ aktif bağlam.

---

## 1. Proje özeti ve mevcut durum

**Amaç:** Sigorta hasar dosyalarına yüklenen araç fotoğraflarında AI üretimi ve
manipülasyon tespiti. İki ayrı problem: (A) tam sentetik görüntü tespiti,
(B) gerçek fotoğraf üzerinde yapılan manipülasyonun tespiti + konumlandırması.

**Hafta 1'de yapılmıştı:** repo iskeleti, `Detector` arayüzü, manifest sistemi,
metrik altyapısı (ROC-AUC, PR-AUC, TPR@FPR, ECE, piksel-F1/IoU), otomatik rapor
motoru, L1 metadata dedektörü, CarDD indirildi (4000 görüntü), E0 sanity deneyi.

**Hafta 2'de bugüne kadar yapılanlar:**

| Katman | Durum | Adet |
|---|---|---|
| R — Gerçek (CarDD) | ✅ | 4.000 |
| R — Kendi telefon fotoğrafları | ✅ | 52 |
| M3 — Klasik manipülasyon | ✅ | 520 |
| S — Tam sentetik | ❌ | 0 |
| M1 — Inpaint hasar ekleme | ❌ | 0 |
| M2 — Inpaint hasar silme | ❌ | 0 |

- **Kendi fotoğraf ingest hattı tamamlandı.** 52 iPhone 13 fotoğrafı, tam EXIF
  (48 tag), GPS ayrıştırıldı, EXIF Orientation normalize edildi. Bu, W1 Bulgu
  4'ü kapattı — L1 metadata katmanını doğrulayabileceğimiz tek veri kaynağı.
- **Maske üretim modülü tamamlandı.** Düzensiz, yumuşak kenarlı manipülasyon
  maskeleri + kabul kapıları.
- **Laundering modülü yazıldı** (5 profil) ama **henüz çalıştırılmadı**.
- **4 üretici modül yazıldı**, sadece `classic_manip` gerçek veride çalıştı.
- **Split hijyeni makine tarafından garanti altında.**

**Manifest durumu:** `data/processed/manifest_v2.parquet`, 4.572 satır.
Gerçek görüntü split dağılımı **2854 / 817 / 381** (train/val/test) — CarDD'nin
kendi küratörlü bölümü + kendi fotoğrafların hash split'i. Bu sayılar sabittir
ve değişmemelidir.

---

## 2. Kod mimarisi ve dosyalar

### Temel akış

```
ham veri  →  build_manifest_v2.py  →  manifest_v2.parquet
                                            ↓
                                    apply_laundering.py
                                            ↓
                              manifest_v2_laundered.parquet
                                            ↓
                          run_e1_shortcut.py / eğitim / değerlendirme
```

**Üreticiler bu akışın yanında durur:** `manifest_v2.parquet`'i OKUR, yeni
görüntü üretir, `gen_log.jsonl` yazar. Manifest'e doğrudan YAZMAZLAR —
manifest yazımı tek noktadan (`build_manifest_v2.py`) yapılır ki doğrulama
adımları atlanamasın.

### src/data/

| Dosya | İşlev | Girdi → Çıktı |
|---|---|---|
| `manifest.py` (W1) | Manifest şeması, `add_row`, `check_split_leakage`, `check_generator_disjoint`, `summarize` | DataFrame ↔ parquet |
| `imageio.py` | **Unicode-güvenli** `imread`/`imwrite`, `orientation_mismatch` | path → ndarray |
| `launder.py` | 5 laundering profili, `launder_file`, `launder_mask` | dosya → dosya |
| `masks.py` | Düzensiz maske üretimi, morfoloji, `vehicle_region` (GrabCut), kalite ölçümleri | ndarray → maske |

**`masks.py` içindeki kritik fonksiyonlar:**
- `mask_for_damage_add` — M1a: mevcut hasarın DIŞINDA bölge seçer
- `mask_for_damage_enlarge` — M1b: halka maskesi (yeni \ eski)
- `mask_for_removal` — M2: hasar maskesi + pay
- `changed_fraction_in_mask` — **kabul kapısı**, maske içinde gerçekten piksel
  değişti mi?
- `leak_fraction_outside_mask` — maske dışına sızıntı ölçer

### src/data/generators/

| Dosya | İşlev |
|---|---|
| `__init__.py` | `GenResult` dataclass — tüm üreticilerin ortak çıktı sözleşmesi |
| `prompts.py` | Kombinatoryal prompt motoru (28.5M kombinasyon), `NEGATIVE_PROMPT` |
| `pipelines.py` | `MODEL_REGISTRY`, difüzyon hattı yükleyicileri, depo yedekleri |
| `fully_synthetic.py` | S katmanı — SD1.5 / SDXL / FLUX ile sıfırdan üretim |
| `inpaint_add.py` | M1 — gerçek fotoğrafa olmayan hasar ekleme / büyütme |
| `inpaint_remove.py` | M2 — hasar silme (sd_inpaint veya OpenCV telea/ns) |
| `classic_manip.py` | M3 — copy-move / splice / bg_replace, **GPU'suz** |

`inpaint_remove.py`, `inpaint_add.py`'den `WORK_SIZE`, `_blend_back`, `_to_work`
ve `_damage_mask_path_from_manifest` import eder. `classic_manip.py` de
`_damage_mask_path_from_manifest`'i oradan alır.

**torch/diffusers import'ları FONKSİYON İÇİNDEDİR.** Bu sayede paket GPU'suz
makinede sorunsuz import edilir; `classic_manip` çalıştırırken torch yüklenmez.

### scripts/

| Dosya | İşlev | Durum |
|---|---|---|
| `inspect_cardd.py` (W1) | CarDD yapısını keşfeder | ✅ |
| `build_manifest_v1.py` (W1) | Hafta 1 manifesti | ✅ |
| `ingest_own_photos.py` | Telefon fotoğrafı ingest: EXIF, GPS ayırma, plaka, yön | ✅ çalıştı |
| `build_manifest_v2.py` | Tüm katmanları birleştirir, split atar, doğrular | ✅ çalıştı |
| `apply_laundering.py` | Laundering kopyalarını üretir | ⏳ **hiç çalıştırılmadı** |
| `run_e0.py` (W1) | E0 sanity deneyi | ✅ |
| `run_e1_shortcut.py` | E1 kestirme-yol teşhisi | ⏳ **hiç çalıştırılmadı** |

### src/detectors/ ve src/eval/ (Hafta 1'den, dokunulmadı)

`base.py` (Detector protokolü + `DetectorOutput`), `metadata.py` (L1),
`cnn_baseline.py`, `metrics.py`, `report.py` (`run_and_report`).

---

## 3. Teknik kararlar ve kurallar

### Kod kuralları (ihlal etme)

1. **`cv2.imread` / `cv2.imwrite` DOĞRUDAN ÇAĞRILMAZ.** Her zaman
   `src.data.imageio.imread/imwrite`. Sebep: proje yolu `Masaüstü` içinde ve
   OpenCV Windows'ta ASCII dışı yolları okuyamıyor.

2. **Kaynak split'leri değişmez.** CarDD split'i klasör yapısından
   (train2017/val2017/test2017), diğerleri id hash'inden gelir. Çatışma
   durumunda **türetilmiş görüntü elenir**, kaynak taşınmaz.

3. **Donör aynı split'ten seçilir.** `splice` ve `bg_replace` iki kaynak
   kullanır; `classic_manip.pick_donor` bunu garanti eder.

4. **`flux_schnell` test-only.** `MODEL_REGISTRY`'de `test_only=True`;
   manipülasyon katmanında kullanılamaz, `build_manifest_v2` bunu doğrular.

5. **Kabul kapısı zorunlu.** Üretilen manipülasyon maske içinde yeterince
   piksel değiştirmiyorsa diske YAZILMAZ (`MIN_CHANGED_IN_MASK`).

6. **Prompt disiplini.** "8k / cinematic / professional" gibi kelimeler
   prompt'a giremez. Her prompt zorunlu olarak bir "amatör kalite" ve bir
   "kamera" özelliği taşır. `prompts.py` bunu test eder.

7. **Maskeler PNG'dir, asla JPEG'lenmez.** JPEG artefaktı ikili maskeyi bozar.

8. **L1 metadata dedektörü ORİJİNAL dosyayı okur**, laundered kopyayı değil.
   Orijinal yol `gen_params["original_path"]` içinde saklanır.

9. **Eğitim ve değerlendirme SADECE laundered kopyalar üzerinden yapılır.**
   Ham dosyalar farklı formatlarda (PNG vs JPEG) — format kestirme yolu.

### Laundering profilleri

| Profil | İşlem | Kalite |
|---|---|---|
| `clean` | yok (sadece yeniden kaydet) | q95 |
| `whatsapp` | uzun kenar 1600 | q75 |
| `screenshot` | 1280 + kırpma + PNG ara adım | q90 |
| `double_jpeg` | q95 → | q70 |
| `aggressive` | 1024 + blur 0.5 | q60 |

Train/val'de sadece `clean, whatsapp, double_jpeg` üretilir; **test'te beşi de**.

### Model seçimi

| Model | Boyut | Rol |
|---|---|---|
| `sd15` | ~4 GB | train/val, T4'te rahat |
| `sdxl` | ~7 GB | ana üretici, T4'te CPU offload ile |
| `flux_schnell` | ~24 GB | **SADECE TEST — T4'e sığmayabilir** |

Depo adresleri yedekli: RunwayML SD1.5 depolarını kaldırdı, `repo_fallbacks`
sırayla denenir.

---

## 4. Sıradaki görevler

### Son push edilenler (commit `f4b95b2`, 19 dosya)

`.gitignore`, `src/data/{launder,masks,imageio}.py`, `src/data/generators/*` (7),
`scripts/{ingest_own_photos,build_manifest_v2,apply_laundering,run_e1_shortcut}.py`,
`notebooks/W2_colab_smoke_test.ipynb`, `data/processed/split_groups.json`,
2 adet `experiments/*/results.json`.

### İlk yapılacaklar (sırayla)

**1. Laundering — hiç çalıştırılmadı**
```powershell
python scripts/apply_laundering.py
```
~14.800 JPEG, 5-6 GB, 20-40 dk. Çıktıdaki "TEST SETI: profil x etiket matrisi"
tablosunda boş hücre olmamalı.

**2. E1 kestirme-yol deneyi — hiç çalıştırılmadı**
```powershell
python scripts/run_e1_shortcut.py
```
Planın "mutlaka yap ve raporla" dediği test. **İlk çalıştırma olduğu için hata
çıkabilir.** Dört prob (meta / px8 / px32 / px32_rgb) × 5 laundering profili
için ROC-AUC üretir. `meta` probu yüksek AUC verirse felaket: model piksel
okumadan sınıflandırabiliyor demektir.

**3. Colab duman testi**
`notebooks/W2_colab_smoke_test.ipynb` — 1. hücrede `REPO_URL`'i
`https://github.com/Tunahan-46/insurance-image-forensics.git` yap.
Gece boyu üretim başlatmadan önce hangi modellerin sığdığını ölçer.

**4. Colab'da S / M1 / M2 üretimi** (gerçekçi hedefler: 300 / 250 / 150)
Inpainting için CarDD verisinin Drive'a yüklenmesi gerekiyor.

**5. Hafta 2 Cuma teslimleri**
- `docs/dataset_card.md` — başlanmadı
- Senaryo başına 3 örnek görsel kolaj — başlanmadı
- Test setini dondur: `python scripts/build_manifest_v2.py --freeze-test`
- `docs/weekly/W2.md` — başlanmadı

---

## 5. Kritik notlar — yeni Claude'un bilmesi gerekenler

**A. OpenCV 5.0.0 kullanılıyor.** `cv2.CascadeClassifier` bu sürümde tamamen
KALDIRILDI. Plaka bulanıklaştırma bu yüzden devre dışı; `ingest_own_photos.py`
bunu zarifçe atlıyor ve kullanıcıyı uyarıyor. Haar cascade kullanan kod yazma.

**B. Proje yolu Türkçe karakter içeriyor** (`Masaüstü`). Kural 1'i unutma.

**C. EXIF Orientation tuzağı.** `cv2.imdecode` yön etiketini UYGULAR, `PIL`
UYGULAMAZ. Bu yüzden `ingest_own_photos.py` kendi fotoğrafları alırken
pikselleri sabitleyip `Orientation=1` yazıyor. Dışarıdan yeni bir görüntü
kaynağı eklenirse aynı normalizasyondan geçirilmeli.

**D. `.gitignore` inceliği.** Git, üst klasörü ignore edilmiş bir dosyayı `!`
ile geri alamaz. Bu yüzden `data/processed/*` yazılı (`data/processed/` değil).
Aynısı `experiments/*` için geçerli.

**E. Veri repoya girmez.** `data/` tamamen gitignore'da. Commit öncesi
`git add -A -n` ile kuru çalıştırma yapıp listeyi kontrol et. Beklenen: sadece
`.py`, `.ipynb`, küçük JSON'lar. Yüzlerce satır çıkıyorsa dur.

**F. KVKK — plakalar HENÜZ bulanıklaştırılmadı.** `data/raw/` gitignore'da
olduğu için araştırma akışında risk yok. Ama şu üç durumdan önce mutlaka
anonimleştirilmeli: veri setini paylaşmadan, sunuma görsel koymadan,
`dataset_card.md` için kolaj üretmeden önce.

**G. Yeniden üretim maliyetli.** `classic_manip` 400 örnek için ~12 dakika,
`apply_laundering` 20-40 dakika. Kullanıcı bu döngülerden yoruldu. Kod
değişikliği önermeden önce gerçekten gerekli olduğundan emin ol; mümkünse
`--splits` / `--limit` gibi kısmi çalıştırma seçeneklerini kullan.

**H. Üreticilerin `resume=True` varsayılanı var.** Mevcut dosyalar atlanır.
Yeniden üretim gerekiyorsa çıktı klasörünü SİLMEK gerekir, yoksa
`gen_log.jsonl` eski kayıtları taşımaya devam eder.

**I. `build_manifest_v2` her çalıştığında split'leri sıfırdan hesaplar.**
Türetilmiş bir manifest'i tekrar girdi olarak vermek güvenlidir (kendini
besleme döngüsü kapatıldı), ama kaynak split'lerinin 2854/817/381'de kaldığını
her seferinde doğrula.

**J. Kullanıcı çalışma tarzı:** Türkçe konuşuyor. Kod değişikliklerini
doğrudan yapıp diske yazmanı bekliyor — her adım için izin isteme. Ama büyük
bir yeniden üretim gerektiren değişikliklerde önce sebebini açıkla. Çıktıları
terminalden kopyalayıp yapıştırıyor; komutları PowerShell sözdizimiyle ver.
