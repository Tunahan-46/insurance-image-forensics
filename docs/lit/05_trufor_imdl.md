# Guillaro vd. — TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization

**Künye:** Fabrizio Guillaro, Davide Cozzolino, Avneesh Sud, Nicholas Dufour,
Luisa Verdoliva · **CVPR 2023** · arXiv:2212.10957

**Problem.** Manipülasyon tespiti (IMDL) yalnızca "sahte mi?" sorusuna
cevap vermemeli; **nerede** sahte olduğunu da göstermeli. Üstelik
konumlandırmanın kendisinin ne kadar güvenilir olduğu da bilinmeli — yanlış
yerde kırmızı boyayan bir ısı haritası, hiç harita olmamasından kötüdür.

**Yöntem.** SegFormer tarzı transformer encoder-decoder. RGB görüntüyü
**Noiseprint++** ile birleştiriyor: yalnızca *gerçek* görüntüler üzerinde
kendi-kendine-denetimli eğitilmiş, kamera/düzenleme artefaktlarını taşıyan
öğrenilmiş gürültü artığı. Üç çıktı veriyor:
1. piksel seviyesinde konumlandırma haritası
2. görüntü seviyesinde bütünlük skoru
3. **güven/güvenilirlik haritası** — konumlandırmaya nerede güvenilmemesi
   gerektiğini işaretliyor

**Veri.** Noiseprint++ eğitimi: **24.757 görüntü, 1.475 kamera modeli**
(43 marka, Flickr/DPReview). Konumlandırma/tespit eğitimi CAT-Net v2 korpusu.
Test: CASIA v1, Coverage, Columbia, NIST16, DSO-1, VIPP (**1.530 sahte +
1.412 gerçek**), OpenForensics (2.000 örnek) + kendi ürettikleri **CocoGlide**
(512 difüzyonla inpaint edilmiş görüntü).

**Metrik ve sonuç.** Ortalama piksel **F1 0.785** (sabit eşik) — CAT-Net v2
0.601, MVSS-Net 0.430. Görüntü seviyesi **AUC 0.857** (0.797 / 0.723).
Dengeli doğruluk **0.781** (0.649 / 0.586).

**Bize yarayan kısım.** Planın P0 listesinde "bir IMDL modelini zero-shot
çalıştır" maddesi var — bu o model. Bizim için özellikle uygun iki sebep:
1. **CocoGlide bizim M1/M2'mizin akrabası.** Difüzyonla inpaint edilmiş
   bölgeler — TruFor'un bu senaryoda ne yaptığı doğrudan bizim asıl
   senaryomuzun cevabı.
2. **Bizde 840 görüntünün 840'ında maske var.** Yani piksel-F1 / IoU
   hesaplayabiliyoruz; TruFor'un çıktısı doğrudan karşılaştırılabilir.
   Görev B için "kendi modelimizi eğitmeden önce hazır SoTA ne veriyor?"
   sorusunun cevabı bu olacak.

**Repo + lisans — engelleyici.** https://github.com/grip-unina/TruFor
(test kodu Haz 2023, eğitim kodu Mar 2025; CocoGlide indirilebilir).
LICENSE.txt: **"This software should be used, reproduced and modified only for
informational and nonprofit purposes."** — **ticari kullanım yasak.**
Staj/araştırma prototipi için sorun yok; ürünleşme senaryosunda ya GRIP-UNINA'dan
lisans alınmalı ya da izinli bir alternatife geçilmeli.

**Açık iş kalemi.** Ticari alternatif adayları **IML-ViT** ve **MVSS-Net** —
lisansları **henüz kontrol edilmedi**. TruFor'un nonprofit şartı bir engel
hâline gelirse ilk yapılacak iş bu.

**Teyit edilemeyen.** Yukarıdaki ortalama F1/AUC değerleri makalenin PDF'inden
tek geçişte çıkarıldı; rapora/sunuma girmeden önce tablodan bir kez daha
doğrulanmalı. Dağıtılan hazır ağırlıkların ayrı bir lisans şartı taşıyıp
taşımadığı da README'de yazmıyor.
