# Cozzolino vd. — Raising the Bar of AI-generated Image Detection with CLIP

**Künye:** Davide Cozzolino, Giovanni Poggi, Riccardo Corvi, Matthias Nießner,
Luisa Verdoliva (GRIP, Napoli Federico II + TUM) · **CVPR 2024 Workshops
(Media Forensics)** — ana track değil · arXiv:2312.00195

**Problem.** Ojha'nın dondurulmuş-CLIP paradigması işe yarıyor, ama pratikte
iki soru açık: (1) ne kadar eğitim verisi gerçekten gerekli? (2) görüntü
sosyal medyadan geçip bozulduğunda ne oluyor?

**Yöntem.** Aynı dondurulmuş CLIP premisi, ama eğitim **eşleştirilmiş**
(paired) kuruluyor: aynı caption'ı paylaşan gerçek/sentetik çiftler, **tek bir**
generator'dan (Latent Diffusion veya ProGAN), gerçekler COCO/LSUN'dan.
Bulgu: 10 çift kadar azıyla bile çalışıyor, en iyi sonuç 1k–10k çiftte.
Yani büyük ve alan-özel bir eğitim seti *ne gerekli ne de pratik*.

**Veri.** **18 generator, ~32.000 gerçek+sahte** görüntülük test paketi:
ProGAN, StyleGAN2/3/-T, GigaGAN, Score-SDE, ADM, GLIDE, eDiff-I, LDM, SD
varyantları, DiT, DeepFloyd-IF + ticari DALL·E 2/3, Midjourney v5, Adobe
Firefly. Hem temiz hem işlenmiş (kırpma, 200×200'e küçültme, JPEG).

**Metrik ve sonuç.** AUC. 10k çiftle **ortalama %89.8 AUC** (augmentation'sız) /
**%90.0** (augmentation'lı). İşlenmiş (laundered) görüntülerde
**%78.1 → %85.2 AUC** — augmentation'ın asıl kazancı burada. Rakiplere göre
ortalama **+%6 AUC** (OOD) ve bozulmuş veride **+%13**.

**Bize yarayan kısım.** İki yerde:
1. **E4 (zero-shot).** Repo hazır ağırlık dağıtıyor; bizim test setimizde hiç
   eğitmeden çalıştırılabilir. Planın "en ilginç sonuç" dediği deney bu —
   dışarıdan gelen bir dedektör bizim araç-hasar alanında ne yapıyor?
2. **E5'in gerekçesi.** Laundering augmentation'ın temiz veride neredeyse hiç
   fark yaratmayıp bozulmuş veride +%7 getirmesi, bizim train'de
   `clean/whatsapp/double_jpeg` karışık eğitme kararımızın literatür dayanağı.

**Repo + lisans.** https://github.com/grip-unina/ClipBased-SyntheticImageDetection
(ağırlıklar `git lfs pull` ile) · **Apache 2.0** — ticari kullanım serbest.
GRIP-UNINA'nın diğer repolarından (bkz. TruFor) bu yönüyle ayrılıyor.

**Teyit edilemeyen.** LFS ile gelen **ağırlıkların** ayrı bir lisans taşıyıp
taşımadığı repo'da açıkça yazmıyor. E4'ü çalıştırmadan önce LICENSE dosyasına
bir kez daha bakılmalı.
