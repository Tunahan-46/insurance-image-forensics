# Wang vd. — CNN-generated images are surprisingly easy to spot... for now

**Künye:** Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens,
Alexei A. Efros (UC Berkeley / Adobe / U. Michigan) · **CVPR 2020** · arXiv:1912.11035

**Problem.** CNN tabanlı generator'lar ortak bir "parmak izi" bırakıyor mu, yoksa
her model kendi izini mi bırakıyor? Pratik soru: tek bir generator'dan eğitilen
dedektör diğerlerini yakalayabilir mi?

**Yöntem.** Süslü bir şey yok — düz bir **ResNet-50** ikili sınıflandırıcı, tek
bir generator'dan (ProGAN) eğitiliyor. Anahtar bileşen **agresif eğitim-zamanı
augmentation'ı**: Gaussian blur + JPEG. Bulgu: bu şekilde eğitilen model diğer
CNN generator'larına şaşırtıcı derecede iyi genelliyor.

**Veri (ForenSynths).** Eğitim: **720k görüntü** (20 LSUN kategorisi × 36k),
gerçek/sahte dengeli. Test: makalede 11 generator (repo README'de 13 test seti,
deepfake ve whichfaceisreal.com dahil); generator başına test boyutu 360 (SITD)
ile 12.800 (CRN/IMLE) arasında.

**Metrik ve sonuç.** Average precision + accuracy. En iyi augmentation'lı
20-sınıf modeli generator ortalamasında **%91.4 mAP**. Generator başına accuracy
%58.6 (DeepFake) ile %100 (ProGAN, in-distribution) arasında.

**Bize yarayan kısım.**
1. **E1/E2'nin tarihsel referansı.** Planın E1'i (ResNet-50 full fine-tune)
   doğrudan bu kurulumun küçültülmüş hali. Ojha'nın makalesi de baseline olarak
   bunu kullanıyor — yani E1 vs E3 karşılaştırmamız literatürdeki
   ana karşılaştırmanın aynısı.
2. **Augmentation fikri.** Bizim `apply_laundering.py`'ımızın atası bu:
   blur + JPEG'i eğitim sırasında enjekte etmek. Fark şu ki biz augmentation'ı
   eğitim döngüsünde değil, veri setinin kendisinde (laundered kopyalar olarak)
   maddileştirdik — böylece **test'te profil başına ayrı raporlama** mümkün
   oluyor, ki plan bunu şart koşuyor.
3. **Başlıktaki "for now" uyarısı bizim için hâlâ geçerli.** Makale GAN
   çağında yazıldı; difüzyon modelleri geldiğinde bu yaklaşımın çöktüğünü Ojha
   gösterdi. Bizim S katmanımız tamamen difüzyon (SD1.5/SDXL/SD-Turbo) — yani
   E1'in düşük çıkması **beklenen** sonuç, başarısızlık değil.

**Repo + lisans — dikkat.** https://github.com/PeterWang512/CNNDetection ·
**CC BY-NC-SA 4.0** (LICENSE.txt). **Ticari olmayan + ShareAlike.** Sigorta
ürününe giden bir hatta bu kod **kullanılamaz**; E1'i sıfırdan kendimiz yazmak
(torchvision ResNet-50 + kendi eğitim döngümüz) ya da Ojha'nın MIT kodunu
kullanmak gerekir.
