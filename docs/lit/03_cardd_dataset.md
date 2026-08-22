# Wang, Li, Wu — CarDD: A New Dataset for Vision-Based Car Damage Detection

**Künye:** Xinkuang Wang, Wenjing Li, Zhongcheng Wu (USTC / PIC Lab) ·
**IEEE T-ITS, cilt 24, sayı 7, s. 7202–7214, 2023** · DOI 10.1109/TITS.2023.3258480 ·
arXiv:2211.00945

**Problem.** Araç hasar tespiti için halka açık, yüksek çözünürlüklü, örnek
seviyesinde anotasyonlu bir set yoktu; alan çoğunlukla kapalı sigorta verisiyle
çalışıyordu.

**İçerik.** **4.000 yüksek çözünürlüklü görüntü, 9.000+ anotasyonlu instance**,
**6 sınıf:** dent (göçük), scratch (çizik), crack (çatlak), glass shatter (cam
kırığı), tire flat (patlak lastik), lamp broken (kırık far). Bölünme:
**train 2.816 / val 810 / test 374**. Ortalama çözünürlük ~684k piksel,
belirtilen minimum 1000×413.

**Metrik ve baseline.** COCO tarzı AP. En iyi baseline DCN+ (ResNet-101):
**box AP 60.6, mask AP 57.0** (AP_small 34.6 / AP_medium 44.0 / AP_large 71.6).
Ayrıca salient object detection (SOD) benchmark'ı da veriliyor.

**Bize yarayan kısım.** Projenin `real` katmanının **tamamı** (4.000/4.052) bu
set. Üç şey doğrudan buradan geliyor:
1. **Split'ler bizde değişmiyor** — CarDD'nin kendi küratörlü bölünmesi
   (train2017/val2017/test2017) aynen korunuyor, biz üstüne kendi kuralımızı
   koymuyoruz. Manifest her çalıştığında 2854/817/381 doğrulanıyor.
2. **SOD maskeleri** M1/M2 katmanının temeli. W1 Bulgu 2'de 4000/4000 eşleşme
   doğrulandı; insan anotasyonlu maske olduğu için SAM ile maske üretme adımı
   tamamen gereksizleşti.
3. **Test setinin küçüklüğü bir kısıt.** 374 görüntü (%9.35) — bu yüzden
   `SPLIT_QUOTA` ile üretilmiş örneklerin **%25'i** test'e yönlendiriliyor,
   yoksa kaynağın doğal dağılımı test'e çok az sentetik bırakırdı (W1'de tam
   olarak bu yaşandı).

**Erişim + lisans — dikkat.** https://cardd-ustc.github.io/ · Açık indirme
**yok**: lisans PDF'i imzalanıp yazarlara e-posta atılıyor, link öyle geliyor.
Şartlar: "istatistiksel ve bilimsel araştırma amaçlı" kullanım, PIC Lab'ın ön
onayıyla; **"herhangi bir ticari kullanım önce PIC Lab tarafından
yetkilendirilmelidir"**; üçüncü taraflara yeniden dağıtım izinsiz yasak;
kurumsal bağlantısı olan gerçek bir kişi imzalamalı.

**Projeye etkisi.** Akademik/staj prototipi için sorun yok. Ancak (a) veri
setimiz **repoda dağıtılamaz** — `data/` gitignore'da, `dataset_card.md` bunu
yazıyor; (b) sigorta şirketine ürünleşme senaryosunda PIC Lab'dan **yazılı izin**
gerekir. Hugging Face'te gayriresmî aynalar var (ör. `harpreetsahota/CarDD`,
2.820 satır) ama **lisans beyanı yok** — otoriter kaynak sayılmamalı.
