# Durum Analizi — Hafta 3 sonu objektif değerlendirme

Bu belge, `docs/mentorluk_plani.md`'nin gerektirdikleri ile deponun **ölçülmüş**
gerçek durumunu karşılaştırır. Rakamlar tahmin değil; `manifest_v2.parquet`,
`manifest_v2_laundered.parquet` ve `experiments/` çıktılarından hesaplandı.

---

## 1. Tek cümlelik teşhis

**Altyapı planın öngördüğünden iyi, veri hacmi planın %30'u, ve planın "asla
kesme" dediği ana deneylerin (E3/E4) hiçbiri henüz çalıştırılmadı.**

Sorun bir hata değil, bir **kaynak dağılımı** sorunu: son iki gün, hedefini
%130 ile aşan tek katmanın (M3 klasik manipülasyon) görsel kalitesine harcandı;
bu sırada ana baseline'ın sonucu hâlâ sıfır.

---

## 2. Veri hacmi — plan hedefi vs gerçek

| Katman | Plan hedefi | Gerçek | Oran | Not |
|---|---:|---:|---:|---|
| R — gerçek | 2.000 | **4.052** | %203 | CarDD 4.000 + kendi fotoğrafın 52 |
| S — tam sentetik | 1.200 | **330** | **%28** | sdxl 80, sd15 150, sd_turbo 100 |
| M1 — inpaint hasar ekleme/büyütme | 800 | **200** | **%25** | add 108 + enlarge 92 |
| M2 — inpaint silme | 400 | **120** | **%30** | inpaint_remove |
| M3 — klasik (copy/splice/bg) | 400 | **520** | **%130** | ✅ tek hedefi aşan katman |
| **Toplam sahte** | **2.800** | **1.170** | **%42** | |
| Laundering profili | 3–5 | **5** | %100 | clean/whatsapp/double_jpeg/screenshot/aggressive |
| Değerlendirme örneği | ~14.400 | **16.966** | %118 | profil çarpımı sayesinde |

**Okunuşu:** Ana baseline'ın (E3, CLIP synthetic detection) beslendiği S katmanı
%28'de. Localization'ın (Hafta 4, E7–E10) beslendiği maskeli havuz 840 örnek —
plan 1.600 diyor. En çok emek verilen M3 ise zaten fazlasıyla yeterliydi.

FLUX üretilmedi → planın E6 tarifi ("train SD1.5+SDXL, test FLUX") harfiyen
uygulanamaz. **Ancak** `sd_turbo` zaten %100 test'e ayrılmış (train 0 / val 0 /
test 100), yani generator-disjoint protokol fiilen kurulmuş durumda. E6 bu
ikame ile dürüstçe çalıştırılabilir; raporda "FLUX yerine SD-Turbo" denmeli.

---

## 3. Ölçülmüş veri kalitesi sorunları

### 3.1 Çözünürlük sızıntısı (Plan §4.5, Tuzak 2) — AÇIK

Tek bir sayı, sadece görüntü genişliği, gerçek ile sentetiği ayırıyor:

| Profil | `width` tek başına AUC | `piksel sayısı` AUC |
|---|---:|---:|
| clean | **0.861** | 0.763 |
| whatsapp | **0.861** | 0.763 |
| double_jpeg | **0.861** | 0.763 |
| aggressive | **0.958** | 0.935 |

Medyan genişlik: gerçek **1000 px**, sentetik **640 px**. Laundering bunu
düzeltmiyor çünkü profiller oranı koruyarak yeniden boyutlandırıyor.

Planın kendi talimatı: *"Tüm görüntüleri aynı boru hattından geçir, çözünürlük
dağılımını eşle."* Bu yapılmadı.

**Etkisi:** CLIP ön-işlemesi 224×224'e indirdiği için bu, E3'ü doğrudan
geçersiz kılmaz; ama yeniden örnekleme artefaktları üzerinden kısmen sızar ve
metadata katmanını (L1) tamamen şişirir. Düzeltilmeden yayınlanacak her Task A
sayısının yanına bu uyarı yazılmak zorunda.

### 3.2 Metadata kestirme yolu — AÇIK ama kısmen meşru

`experiments/E1_shortcut/table.md`:

| Görev | meta | px8 | px32 | px32_rgb |
|---|---:|---:|---:|---:|
| A (sentetik) | **0.976–0.988** | 0.778 | 0.647 | 0.716 |
| B (manipüle) | 0.518–0.541 | 0.499 | 0.478 | 0.456 |

- **Task B temiz.** Doğrulandı: çözünürlükle ayırt etme AUC'si 0.506. Bu
  katmanda kestirme yol yok — laundering işini yapmış.
- **Task A'da meta 0.98.** Sebebi: gerçekler EXIF'li kamera JPEG'i, üretilenler
  EXIF'siz. Üretimde bu **meşru bir sinyaldir** (planın L1 katmanı tam da bunun
  için var) — ama bizim verimizde gerçek/sahte ile %100 örtüştüğü için
  "genelleşiyor" iddiası kurulamaz. Dürüst çözüm: L1'i ayrı raporlamak ve
  füzyonda ağırlığını sınırlamak.
- **px32 = 0.647.** Planın alarm eşiği %95; altındayız. Yani piksel seviyesinde
  felaket bir sızıntı yok. Bu iyi haber.

### 3.3 Dondurulmuş test seti — BOZULDU

`test_manifest_frozen.parquet` = 755 satır. Güncel manifest'in test bölümü =
650 satır. Ortak 608; frozen'da olup artık var olmayan **147**; yeni gelen 42.
`test_manifest_frozen.sha256` **boş**.

M3'ü yeniden ürettiğimiz için bu kaçınılmazdı ve **henüz hiçbir model
eğitilmediği için zararsız** — ama yeniden dondurulması şart, ve ilk E3
koşusundan **önce** yapılmalı.

---

## 4. Kod ve deney disiplini

| Planın istediği | Durum |
|---|---|
| `src/{data,detectors,fusion,eval,api}` iskeleti | ✅ hepsi mevcut |
| `pytest` geçen 5–10 test | ✅ **13 test** |
| `manifest.py` (parquet, klasöre güvenme) | ✅ |
| `metrics.py` (AUC, PR-AUC, TPR@FPR, ECE, piksel F1/IoU) | ✅ |
| `metadata.py` (EXIF, quant tablo, C2PA) | ✅ |
| 5 üretici modül + `launder.py` | ✅ 6/6 |
| `clip_embed.py` (cache'li) | ✅ yazıldı, **hiç çalıştırılmadı** |
| `clip_probe.py` | ✅ yazıldı, **hiç çalıştırılmadı** |
| `calibration.py` | ✅ yazıldı, kullanılmadı |
| `dataset_card.md` | ✅ |
| `docs/lit/` 6 makale notu | ✅ lisans analiziyle |
| Görsel kolaj (senaryo başına 3 örnek) | ✅ 7 kolaj |
| **MLflow / W&B** (plan: "ZORUNLU") | ❌ hiç kullanılmıyor |
| **3 seed × ortalama ± std** | ❌ tek seed |
| `experiments/EXX/{config.yaml,run.py,NOTES.md}` | ❌ sadece results.json |
| İnsan baseline'ı (20+20 görüntü) | ❌ yapılmadı |
| `gradcam.py` | ❌ yok |

### Çalıştırılmış deneyler

| Deney | Plan haftası | Durum | Sonuç |
|---|---|---|---|
| E0 sanity (ResNet-50) | 1 | ✅ koştu | AUC 0.36 — tasarımı gereği anlamsız |
| E0 metadata | 1 | ✅ koştu | AUC 0.50 |
| E1_shortcut (teşhis) | — | ✅ koştu | planda yok, bizim eklediğimiz teşhis |
| **E1** ResNet-50 full FT | 3 | ❌ | — |
| **E2** EfficientNet/ConvNeXt | 3 | ❌ | — |
| **E3 CLIP+LR (ANA BASELINE)** | 3 | ❌ | — |
| **E4** ClipBased zero-shot | 3 | ❌ | — |
| **E5** laundering augmentation | 3 | ❌ | — |
| **E6** generator-disjoint | 3 | ❌ | — |

**Hafta 3'ün teslimi (6 deneyin karşılaştırma tablosu + ROC eğrileri +
kalibrasyon eğrisi + seçilmiş model): %0.**

---

## 5. Hafta 1 / 2 / 3 — yapılanlar ve kalanlar

### Hafta 1 — %95 tamam

| Görev | Durum |
|---|---|
| Repo iskeleti | ✅ |
| `base.py` / `manifest.py` / `metrics.py` / `report.py` / `metadata.py` | ✅ |
| 5–10 pytest | ✅ 13 |
| E0 + ROC PNG | ✅ |
| `docs/weekly/W1.md` | ✅ |
| 6 makale literatür notu | ✅ |
| Kendi telefon fotoğrafın 50–150 | ✅ 52 |
| MLflow kurulumu | ❌ |

### Hafta 2 — %65 tamam

| Görev | Durum |
|---|---|
| 5 üretici modül + launder | ✅ 6/6 |
| Prompt motoru + "amatör" disiplini + negative prompt | ✅ |
| Maskeler (dikdörtgen değil, yumuşak kenarlı) | ✅ (bu hafta düzeltildi) |
| Sızıntısız split (source-image-disjoint) | ✅ |
| Generator-disjoint | ✅ sd_turbo held-out |
| 5 laundering profili | ✅ |
| `dataset_card.md` | ✅ |
| Görsel kolaj | ✅ |
| **~4.800 görüntü hedefi** | ❌ **1.170 sahte (%42)** |
| **Test setini dondur** | ⚠️ donduruldu, sonra bozuldu — yenilenmeli |
| **Çözünürlük dağılımını eşle (Tuzak 2)** | ❌ **açık, AUC 0.861** |
| Kalite kontrolü: 100 görüntüyü gözle incele | ⚠️ kolajla kısmen |
| İnsan baseline'ı (20+20) | ❌ |

### Hafta 3 — %15 tamam

| Görev | Durum |
|---|---|
| `clip_embed.py` (cache'li) | ✅ yazıldı |
| `clip_probe.py` (grid search + threshold + Platt) | ✅ yazıldı |
| `calibration.py` | ✅ yazıldı |
| Kaggle embedding notebook'u | ✅ yazıldı |
| **CLIP embedding çıkarımı** | ❌ çalıştırılmadı |
| **E1–E6** | ❌ 0/6 |
| **6 deneyin karşılaştırma tablosu** | ❌ |
| **ROC eğrileri tek grafikte** | ❌ |
| **Kalibrasyon eğrisi** | ❌ |
| **Seçilmiş model + kaydedilmiş threshold** | ❌ |
| `gradcam.py` | ❌ |
| `docs/weekly/W3.md` | ✅ (ama deney sonucu içermiyor) |

---

## 6. Neyin garantisi yok — ve alternatifi

| Risk | Gerçekçi mi? | Alternatif |
|---|---|---|
| 4.800 görüntülük hedefe ulaşmak | ❌ Hayır — 1.630 görüntü daha üretmek 2+ tam Kaggle oturumu | **Hedefi 2.000'e çek.** Plan P0'da hacim değil, *doğru split + CLIP baseline* var. S'yi 330→600'e çıkarmak E3 için yeterli. |
| 17 deneyin tamamı | ❌ Hayır | **8 deney hedefle:** E0(var), E3, E4, E5, E6, E7, E12, E16. Planın P0+P1 listesi zaten bu. |
| FLUX ile generator-disjoint | ❌ FLUX üretilmedi | ✅ **sd_turbo zaten held-out** — E6 bununla koşulur, farkı raporlanır |
| E10 (IMDL fine-tune) | ❌ | Plan zaten **P2 = kesilebilir** diyor. Kes. |
| MLflow'u geriye dönük kurmak | ⚠️ mümkün ama zaman kaybı | `results.json` + `NOTES.md` konvansiyonunu sürdür, sapmayı raporda dürüstçe yaz |
| FastAPI + Docker + Gradio | ⚠️ Gradio P0, diğerleri P1 | **Önce Gradio.** Docker zaman kalırsa. |
| 3 seed × ± std | ✅ ucuz (linear probe saniyeler sürer) | E3 için mutlaka yap |

---

## 7. Önerilen sıra (bundan sonraki 5 iş)

1. **Çözünürlük eşitleme + yeniden dondurma.** Tüm katmanları tek boru hattından
   aynı uzun-kenar dağılımıyla geçir, `build_manifest_v2.py --freeze-test`
   çalıştır, sha256'yı dosyaya yaz. *Bu, E3'ten önce yapılmazsa tüm Task A
   sonuçları şüpheli doğar.*
2. **CLIP embedding çıkarımı** (Kaggle T4, notebook hazır).
3. **E3 + E4 + E5 + E6** — hepsi aynı embedding cache'i kullanır, dakikalar sürer.
   3 seed ile koş.
4. **Hafta 3 teslimini kapat:** karşılaştırma tablosu, tek grafikte ROC'lar,
   kalibrasyon eğrisi, seçilmiş model + threshold, `W3.md`'ye "Bulgu: X".
5. **Ancak ondan sonra** S katmanını 330→600'e çıkarmayı değerlendir.

---

## 8. Dürüst kapanış

İyi olan: altyapı kalitesi planın üstünde, split disiplini gerçek ve
doğrulanmış, Task B kestirme yoldan arınmış, dokümantasyon (dataset card,
literatür, kolaj) planın istediği biçimde mevcut. Bunlar geri alınamaz kazanım.

Kötü olan: üç haftanın sonunda **ölçülmüş tek bir model sonucu yok**. Planın
"asla kesme" listesinin ilk maddesi olan CLIP baseline'ı hâlâ çalıştırılmadı.
Son iki günün emeği gerçek bir kalite iyileştirmesiydi ama en az kritik
katmandaydı ve zamanlaması yanlıştı.

Kurtarılabilir mi: evet. E3'ün kendisi bir saatlik iş — çünkü kodu zaten yazıldı
ve test edildi. Eksik olan sadece embedding çıkarımı ve bir koşu.
