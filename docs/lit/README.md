# Literatür notları

Plan Hafta 1 teslimi: 6 makale, her biri için problem / yöntem / veri / metrik /
bize yarayan kısım / repo + lisans.

| # | Not | Çalışma | Neden okundu |
|---|---|---|---|
| 1 | [`01_ojha_universal_fake_detect.md`](01_ojha_universal_fake_detect.md) | Ojha vd., CVPR 2023 | **Ana baseline** — E3'ün doğrudan reçetesi |
| 2 | [`02_cozzolino_clip_bar.md`](02_cozzolino_clip_bar.md) | Cozzolino vd., CVPRW 2024 | E4 zero-shot; laundering'e dayanıklılık |
| 3 | [`03_cardd_dataset.md`](03_cardd_dataset.md) | Wang vd., T-ITS 2023 | Projenin `real` katmanının tamamı |
| 4 | [`04_wang_cnndetection.md`](04_wang_cnndetection.md) | Wang vd., CVPR 2020 | E1/E2'nin tarihsel referansı; augmentation fikri |
| 5 | [`05_trufor_imdl.md`](05_trufor_imdl.md) | Guillaro vd., CVPR 2023 | Görev B: manipülasyon **konumlandırma** |
| 6 | [`06_wu_osn_robustness.md`](06_wu_osn_robustness.md) | Wu vd., CVPR 2022 | Laundering katmanının literatür dayanağı |

## Lisans özeti — projeye doğrudan etkisi

| Çalışma | Lisans | Ticari kullanım |
|---|---|---|
| UniversalFakeDetect (Ojha) | MIT | ✅ serbest |
| ClipBased (Cozzolino) | Apache 2.0 | ✅ serbest |
| ImageForensicsOSN (Wu) | MIT | ✅ serbest |
| CNNDetection (Wang) | CC BY-NC-SA 4.0 | ❌ **ticari kullanım yasak** |
| TruFor (Guillaro) | "nonprofit purposes only" | ❌ **ticari kullanım yasak** |
| CarDD | İmzalı form + PIC Lab izni | ⚠️ ticari kullanım ayrı izin ister |

**Sonuç:** Prototip akademik/araştırma amaçlı olduğu sürece hepsi kullanılabilir.
Sigorta şirketine ürünleşecek bir hat kurulacaksa CNNDetection ve TruFor kodları
**kullanılamaz** — ikisinin de yerine geçebilecek izinli alternatif gerekir
(Ojha MIT'tir; IMDL tarafında IML-ViT / MVSS-Net lisansları **henüz kontrol
edilmedi**, bu açık bir iş kalemi).

---

**Doğrulama notu:** Tüm künye, sayı ve lisans bilgileri Ağustos 2026'da
kaynakların kendisinden (arXiv / CVF açık erişim / GitHub) teyit edildi.
Her notun sonunda teyit edilemeyen kalemler ayrıca işaretlidir.
