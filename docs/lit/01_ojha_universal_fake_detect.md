# Ojha, Li, Lee — Towards Universal Fake Image Detectors that Generalize Across Generative Models

**Künye:** Utkarsh Ojha, Yuheng Li, Yong Jae Lee (UW–Madison) · **CVPR 2023** · arXiv:2302.10174

**Problem.** Gerçek-sahte sınıflandırıcıları eğitildikleri generator'ın
artefaktına "asimetrik olarak" kilitleniyor: görülmemiş bir generator'ın çıktısı
sistematik olarak *gerçek* tarafına düşüyor. Yani in-distribution %99 alan bir
model, yeni bir difüzyon modeli karşısında şansa iniyor.

**Yöntem.** Uçtan uca eğitim yok. **Dondurulmuş CLIP ViT-L/14** görüntü
kodlayıcısı kullanılıyor ve sınıflandırma bu *sabit* özellik uzayında yapılıyor:
(a) gerçek/sahte özellik bankasına karşı en-yakın-komşu, veya (b) tek bir
lineer prob. Özellik uzayı hiç fine-tune edilmiyor — kritik nokta bu.

**Veri.** Eğitim: Wang vd.'nin ProGAN seti, **720k görüntü** (360k gerçek /
360k sahte), 20 LSUN kategorisi. Test: **19 generator** — GAN ailesi (ProGAN,
StyleGAN, BigGAN, CycleGAN, StarGAN, GauGAN, CRN, IMLE, SAN, SITD, DeepFakes),
difüzyon (Guided/ADM, 3× LDM, 3× GLIDE) ve otoregresif (DALL·E).

**Metrik ve sonuç.** mAP + accuracy@0.5. Görülmemiş difüzyon/AR modellerinde
önceki en iyiye göre **+15.07 mAP** ve **+%25.90 accuracy**. Lineer prob (LC)
bu grupta **95.00 mAP** (Wang vd. baseline: 75.51). NN (k=9) ortalama
**%84.25** accuracy (baseline %76.26).

**Bize yarayan kısım.** Bu makale planın E3 deneyinin **birebir reçetesi**:
CLIP ViT-L/14 dondur → 768-d embedding çıkar ve cache'le → `LogisticRegression`
→ threshold seç → kalibre et. Projedeki asıl değeri şu: bizim veri setimiz
küçük (330 sentetik), uçtan uca fine-tune için fazlasıyla küçük. Dondurulmuş
özellik + lineer prob tam da bu rejim için tasarlanmış. Ayrıca E6'nın
(generator-disjoint, `sd_turbo` test-only) hipotezi doğrudan bu makaleden
geliyor: lineer probun görülmemiş generator'a CNN baseline'dan daha iyi
genellemesi bekleniyor.

**Repo + lisans.** https://github.com/WisconsinAIVision/UniversalFakeDetect ·
**MIT** — ticari kullanım dahil serbest. Backbone OpenAI CLIP ViT-L/14 (MIT).

**Uyarı.** Eğitim seti 720k; bizde 1.170 sahte var. Mutlak AUC rakamları
karşılaştırılabilir değil, karşılaştırılabilir olan **göreli davranış**
(lineer prob vs fine-tune, görülen vs görülmemiş generator).
