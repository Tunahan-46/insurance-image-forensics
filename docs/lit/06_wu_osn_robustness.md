# Wu vd. — Robust Image Forgery Detection Over Online Social Network Shared Images

**Künye:** Haiwei Wu, Jiantao Zhou, Jinyu Tian, Jun Liu (University of Macau) ·
**CVPR 2022, s. 13440–13449**

**Problem.** Forensic dedektörler laboratuvarda çalışıyor, sahada çöküyor.
Sebep: gerçek dünyada görüntü sosyal ağdan geçiyor — yeniden sıkıştırılıyor,
boyutlandırılıyor, filtreleniyor. Manipülasyonun bıraktığı ince iz, platformun
bıraktığı izin altında kayboluyor.

**Yöntem.** OSN bozulmasını **iki bileşene** ayırıyorlar:
- *Öngörülebilir* gürültü: JPEG yeniden sıkıştırma / yeniden boyutlandırma.
  Türevlenebilir bir JPEG katmanı içeren U-Net ile öğreniliyor.
- *Görülmemiş* gürültü: OSN artığı bütçesiyle sınırlandırılmış çekişmeli
  (adversarial) bozulma.

İkisi de eğitim sırasında enjekte ediliyor; böylece konumlandırıcı platform
bozulmasına karşı değişmez (invariant) hâle geliyor.

**Veri.** Eğitim: Dresden (bozulmamış) + MS-COCO (nesne yapıştırma) + OSN
işlemlerini öğrenmek için ~1.300 görüntülük bir set. Test: DSO, Columbia, NIST,
CASIA'dan kurulmuş **5.232 sahte + maske**, her biri **Facebook, WeChat, Weibo**
üzerinden geçirilmiş.

**Metrik ve sonuç.** AUC / F1 / IoU. Facebook'tan geçmiş görüntülerde
**AUC 0.847, F1 0.488, IoU 0.400** — baseline 0.694 / 0.331 / 0.259.
Yani laundering'e özel eğitim, konumlandırmada F1'i yaklaşık **%50 artırıyor**.

**Bize yarayan kısım.** Bu makale, projedeki laundering katmanının **var oluş
gerekçesi**. Üç somut bağlantı:
1. **`whatsapp` / `screenshot` profilleri keyfi değil.** Wu vd. tam olarak bu
   kanalları modelliyor. Bizim profillerimiz (1600px+q75, 1280+PNG ara adım)
   aynı fikrin araç-hasar senaryosuna uyarlanmış hâli.
2. **Test'te profil başına ayrı raporlama şart.** Makale gösteriyor ki tek bir
   ortalama AUC, kanal bazındaki çöküşü gizliyor. Planın "sadece `clean`
   üzerinde ölçme" yasağının ampirik dayanağı bu.
3. **E5 için doğrudan hipotez.** Laundering augmentation'lı eğitim, laundered
   test verisinde belirgin kazanç vermeli — Wu vd.'de F1 0.331 → 0.488,
   Cozzolino vd.'de AUC %78.1 → %85.2. İki bağımsız çalışma aynı yönü
   gösteriyor; bizim E5'te de aynı yönü görmemiz beklenir.

**Repo + lisans.** https://github.com/HighwayWu/ImageForensicsOSN ·
Eğitilmiş dedektör ağırlıkları + OSN-gürültü modeli (Facebook) Google Drive /
Baidu Pan üzerinden; topladıkları OSN'den geçmiş veri seti de yayınlanmış ·
**MIT** — ticari kullanım serbest.

**Teyit edilemeyen.** Değerlendirmenin bir kısmında **WhatsApp**'ın da yer
aldığı bilgisi ikincil kaynaktan geldi; makalede birincil olarak Facebook,
WeChat, Weibo geçiyor. WhatsApp'a atıf yapılacaksa makaleden doğrulanmalı.
Lisans bilgisi GitHub sayfası altbilgisinden okundu, LICENSE dosyasının ham
metninden değil.
