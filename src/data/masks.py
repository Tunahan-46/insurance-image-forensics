"""
Manipulasyon maskesi uretimi (plan Hafta 2, "Inpainting senaryosu").

NEDEN AYRI BIR MODUL
--------------------
Plan acikca uyariyor: "Maskeyi tam dikdortgen yapma. Dikdortgen maske ->
model kenarlari ezberler -> sahte yuksek performans." Maske uretimi bu
projede bir yardimci fonksiyon degil, DENEYIN GECERLILIGINI belirleyen
bir bilesendir. Bu yuzden kendi modulu, kendi testleri var.

W1 BULGUSU
----------
CarDD'nin 4000 goruntusunun tamami icin insan anotasyonlu HASAR maskesi
mevcut (CarDD_SOD). Yani SAM adimina gerek yok. Ama dikkat:

    CarDD maskesi = HASARIN NEREDE OLDUGU
    bizim maskemiz = NEREYI DEGISTIRDIGIMIZ

Bunlar farkli seylerdir ve iki farkli senaryoda farkli kullanilir:

  M1 (hasar EKLEME)  : hasarsiz bir bolgeye hasar ekleriz.
                       -> CarDD maskesinin DISINDA bir bolge secilir.
                       -> Aksi halde "zaten hasarli yere hasar ekleme"
                          olur, gercekci degildir.
  M1b (hasar BUYUTME): mevcut CarDD maskesi genisletilir (dilate),
                       halka bolgesi (yeni \\ eski) inpaint edilir.
  M2 (hasar SILME)   : dogrudan CarDD maskesi kullanilir (biraz genisletilmis).

Uc durumda da cikti maskesi, dedektorun bulmasi beklenen ZEMIN GERCEGIDIR.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Uretilen maskeler her zaman uint8 {0,255}, tek kanal.
MASK_FG = 255
MASK_BG = 0


# ---------------------------------------------------------------------------
# Temel sekiller
# ---------------------------------------------------------------------------


def random_blob(
    height: int,
    width: int,
    *,
    rng: np.random.Generator,
    center: tuple[int, int] | None = None,
    radius_frac: float = 0.12,
    n_vertices: int = 12,
    irregularity: float = 0.45,
) -> np.ndarray:
    """Duzensiz, kapali bir cokgen blob uretir (serbest el secimi taklidi).

    Yontem: merkez etrafinda esit acilarla yerlestirilen n_vertices nokta,
    her birinin yaricapi rastgele carpanla bozulur, sonra cokgen doldurulur.
    Dikdortgen/elips yerine bunu kullanmamizin sebebi plan 4.5 Tuzak:
    duzenli kenar = ogrenilebilir kestirme yol.
    """
    cy, cx = center if center else (height // 2, width // 2)
    base_r = radius_frac * min(height, width)

    angles = np.sort(rng.uniform(0, 2 * np.pi, n_vertices))
    radii = base_r * (1.0 + rng.uniform(-irregularity, irregularity, n_vertices))
    radii = np.clip(radii, base_r * 0.25, base_r * 1.8)

    pts = np.stack(
        [cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1
    ).astype(np.int32)

    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [pts], MASK_FG)
    return mask


def feather(mask: np.ndarray, blur_px: int = 9, threshold: int = 110) -> np.ndarray:
    """Kenarlari yumusatip yeniden esikler. Sonuc yine IKILI'dir ama sinir
    cizgisi artik piksel-mukemmel duz degil, dalgalidir.

    Neden yeniden esikliyoruz: zemin gercegi maskesi ikili olmali, aksi
    halde piksel-F1 esik secimine bagimli hale gelir ve raporlanan sayi
    manipule edilebilir olur. Yumusatmayi INPAINT'e verirken (blend icin)
    ayrica `soft_mask` ile aliriz.
    """
    k = blur_px if blur_px % 2 == 1 else blur_px + 1
    blurred = cv2.GaussianBlur(mask, (k, k), 0)
    return ((blurred > threshold).astype(np.uint8)) * MASK_FG


def soft_mask(mask: np.ndarray, blur_px: int = 15) -> np.ndarray:
    """Inpaint pipeline'a verilecek YUMUSAK maske (0-255 arasi degerler).
    Sadece harmanlama icindir; manifest'e YAZILMAZ."""
    k = blur_px if blur_px % 2 == 1 else blur_px + 1
    return cv2.GaussianBlur(mask, (k, k), 0)


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, k)


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask, k)


def roughen(mask: np.ndarray, rng: np.random.Generator, strength: int = 7) -> np.ndarray:
    """Rastgele erozyon/dilatasyon zinciriyle sinirlari dogallastirir."""
    out = mask
    for _ in range(3):
        px = int(rng.integers(1, max(2, strength)))
        out = dilate(out, px) if rng.random() < 0.5 else erode(out, px)
    return feather(out, blur_px=int(rng.integers(5, 15)))


def mask_area_frac(mask: np.ndarray) -> float:
    return float((mask > 127).sum()) / float(mask.size)


def changed_fraction_in_mask(
    original_bgr: np.ndarray,
    manipulated_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    tol: int = 8,
) -> float:
    """Maske icindeki piksellerin yuzde kaci GERCEKTEN degisti?

    NEDEN VAR: Bir manipulasyon hattinin sessizce basarisiz olmasi cok
    kolaydir -- kopyalanan bolge kaynagiyla ayni renkte olabilir, inpaint
    modeli maskeyi neredeyse aynen yeniden uretebilir, seamlessClone
    tamamen notrlestirebilir. Bu durumda diske "manipule" etiketli ama
    aslinda ORIJINALLE AYNI bir goruntu ve yaninda "burasi manipule"
    diyen bir maske yazariz.

    Bu, veri setine zehir enjekte etmektir: piksel-F1 asla 1'e yaklasamaz
    cunku zemin gercegi yalandir, ve goruntu-seviyesi etiket de yanlistir.
    Uretim hatlari bu orani bir KABUL KAPISI olarak kullanir.
    """
    diff = np.abs(original_bgr.astype(np.int16) - manipulated_bgr.astype(np.int16))
    changed = diff.max(axis=2) > tol
    inside = mask > 127
    n = int(inside.sum())
    if n == 0:
        return 0.0
    return float((changed & inside).sum()) / float(n)


def leak_fraction_outside_mask(
    original_bgr: np.ndarray,
    manipulated_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    tol: int = 8,
) -> float:
    """Maske DISINDA degisen piksel orani.

    Sifira yakin olmali. Yuksekse zemin gercegi maskesi eksik demektir
    (orn. VAE round-trip tum goruntuyu degistirmis, bkz. inpaint_add.
    _blend_back). Localization metriginin gecerliligi buna baglidir."""
    diff = np.abs(original_bgr.astype(np.int16) - manipulated_bgr.astype(np.int16))
    changed = diff.max(axis=2) > tol
    outside = mask <= 127
    n = int(outside.sum())
    if n == 0:
        return 0.0
    return float((changed & outside).sum()) / float(n)


# ---------------------------------------------------------------------------
# Arac / hasar bolgesi secimi
# ---------------------------------------------------------------------------


# GrabCut'in calisacagi maksimum uzun kenar (bkz. vehicle_region).
GRABCUT_MAX_DIM = 512


def vehicle_region(image_bgr: np.ndarray, *, max_dim: int = GRABCUT_MAX_DIM) -> np.ndarray:
    """Kaba bir "arac govdesi" maskesi.

    SAM kullanmiyoruz cunku W1'de CarDD maskelerinin hazir oldugu goruldu;
    buradaki amac sadece "gokyuzu/asfalt yerine arac uzerine denk gelsin"
    kisitini saglamak. GrabCut, merkez dikdortgeni tohum alarak bunu
    GPU'suz yapar.

    NEDEN KUCULTUYORUZ -- OLCULDU
    -----------------------------
    GrabCut'in maliyeti piksel sayisiyla dogru orantili:

        1000x1000  ( 1.0 MP)   1.8 sn
        4032x3024  (12.2 MP)  31.3 sn      <- telefon fotograflari

    Ilk surumde tam cozunurlukte calisiyordu ve 400 ornek uretmek ~110
    dakika suruyordu. Oysa bu fonksiyonun urettigi maske ZATEN KABA bir
    bolge tahmini: "bu nokta aracin uzerinde mi?" sorusuna cevap veriyor.
    Bunun icin 12 megapiksel islemek gereksiz.

    Cozum: GrabCut kucultulmus kopyada calisir, sonuc maskesi orijinal
    boyuta buyutulur. Buyutme INTER_LINEAR + esikleme ile yapilir; NEAREST
    merdiven basamagi seklinde kenarlar birakir ve bg_replace senaryosunda
    bu maske ZEMIN GERCEGI oldugu icin duzenli basamaklar modelin
    ezberleyebilecegi yapay bir iz olurdu (plan 4.5 mantigi).

    GrabCut basarisiz olursa merkez %60'lik bolgeye duser -- sessizce
    yanlis sonuc uretmek yerine bilinen bir geri cekilme davranisi.
    """
    H, W = image_bgr.shape[:2]

    scale = min(1.0, max_dim / max(H, W))
    if scale < 1.0:
        small = cv2.resize(image_bgr, (max(32, int(W * scale)), max(32, int(H * scale))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = image_bgr
    h, w = small.shape[:2]

    rect = (int(w * 0.08), int(h * 0.08), int(w * 0.84), int(h * 0.84))
    try:
        gc_mask = np.zeros((h, w), np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(small, gc_mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), MASK_FG, 0)
        out = out.astype(np.uint8)
        if mask_area_frac(out) > 0.05:
            if out.shape != (H, W):
                out = cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)
                out = ((out > 127).astype(np.uint8)) * MASK_FG
            return out
    except cv2.error:
        pass

    fallback = np.zeros((H, W), np.uint8)
    fallback[int(H * 0.2) : int(H * 0.8), int(W * 0.2) : int(W * 0.8)] = MASK_FG
    return fallback


def sample_point_in(mask: np.ndarray, rng: np.random.Generator) -> tuple[int, int] | None:
    """Maske icinden rastgele bir (y, x) noktasi secer."""
    ys, xs = np.nonzero(mask > 127)
    if len(ys) == 0:
        return None
    i = int(rng.integers(0, len(ys)))
    return int(ys[i]), int(xs[i])


# ---------------------------------------------------------------------------
# Senaryo bazli maske uretimi
# ---------------------------------------------------------------------------


def mask_for_damage_add(
    image_bgr: np.ndarray,
    damage_mask: np.ndarray | None,
    *,
    rng: np.random.Generator,
    radius_frac: float = 0.11,
    min_area_frac: float = 0.004,
    max_area_frac: float = 0.10,
) -> np.ndarray | None:
    """M1: HASARSIZ bir panele hasar eklemek icin bolge sec.

    Kisitlar:
      - Arac govdesi icinde olmali (gokyuzune cizik atmak anlamsiz)
      - Mevcut hasar maskesinin DISINDA olmali (+20px guvenlik payi)
      - Alan orani makul olmali; cok kucuk = tespit edilemez, cok buyuk =
        gercekci degil ve "yari sentetik goruntu"ye donusur

    Uygun bolge bulunamazsa None doner -- KAYNAK GORUNTU ATLANIR.
    Zorlayarak kotu maske uretmek, veri setine gurultu sokar.
    """
    h, w = image_bgr.shape[:2]
    body = vehicle_region(image_bgr)

    allowed = body.copy()
    if damage_mask is not None:
        allowed = cv2.bitwise_and(allowed, cv2.bitwise_not(dilate(damage_mask, 20)))
    # Kenara tasan bloblari engelle
    allowed = erode(allowed, int(radius_frac * min(h, w) * 0.8))

    for _ in range(25):
        pt = sample_point_in(allowed, rng)
        if pt is None:
            return None
        blob = random_blob(
            h, w, rng=rng, center=pt,
            radius_frac=radius_frac * float(rng.uniform(0.7, 1.3)),
            n_vertices=int(rng.integers(8, 16)),
        )
        blob = roughen(blob, rng)
        blob = cv2.bitwise_and(blob, body)  # aracin disina tasmasin
        area = mask_area_frac(blob)
        if min_area_frac <= area <= max_area_frac:
            return blob
    return None


def mask_for_damage_enlarge(
    damage_mask: np.ndarray, *, rng: np.random.Generator, grow_px: int | None = None
) -> np.ndarray | None:
    """M1b: Mevcut hasari BUYUTME. Zemin gercegi = yeni alan \\ eski alan
    (halka). Cunku eski hasar bolgesi gercek, sadece halka manipuledir.

    Bu, plan 4.2'deki "hasar buyutme" senaryosunun tam karsiligidir ve
    tablo 4.1'e gore tespiti EN ZOR saldiridir."""
    if mask_area_frac(damage_mask) < 0.001:
        return None
    if grow_px is None:
        grow_px = int(rng.integers(12, 40))
    grown = roughen(dilate(damage_mask, grow_px), rng)
    ring = cv2.bitwise_and(grown, cv2.bitwise_not(dilate(damage_mask, 2)))
    if mask_area_frac(ring) < 0.002:
        return None
    return ring


def mask_for_removal(
    damage_mask: np.ndarray, *, rng: np.random.Generator, pad_px: int | None = None
) -> np.ndarray | None:
    """M2: Hasari SILME. Silinen bolge = hasar maskesi + pay.
    Pay sart: inpaint modelinin hasarin kenar golgelerini de temizlemesi
    gerekir, aksi halde silinmis hasarin haleleri kalir."""
    if mask_area_frac(damage_mask) < 0.001:
        return None
    if pad_px is None:
        pad_px = int(rng.integers(8, 25))
    return roughen(dilate(damage_mask, pad_px), rng)


# ---------------------------------------------------------------------------
# G/C yardimcilari
# ---------------------------------------------------------------------------


def load_mask(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Maskeyi (H, W) uint8 {0,255} olarak yukler. size=(W, H) verilirse
    NEAREST ile yeniden boyutlandirir."""
    m = np.array(Image.open(path).convert("L"))
    if size is not None and (m.shape[1], m.shape[0]) != size:
        m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    return ((m > 127).astype(np.uint8)) * MASK_FG


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(path, format="PNG")


def empty_mask_like(image_bgr: np.ndarray) -> np.ndarray:
    """Real / fully_synthetic goruntuler icin tum-0 maske.
    Localization metrikleri (piksel-F1) bunlari 'manipule bolge yok'
    olarak degerlendirir; None gecmekten daha az ozel-durum kodu gerektirir."""
    return np.zeros(image_bgr.shape[:2], np.uint8)


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Sentetik 'arac': koyu arka plan uzerinde acik renkli bir govde.
    img = np.full((720, 1080, 3), 40, np.uint8)
    cv2.ellipse(img, (540, 400), (380, 180), 0, 0, 360, (190, 185, 180), -1)
    cv2.rectangle(img, (350, 250), (730, 400), (170, 165, 160), -1)

    dmg = np.zeros((720, 1080), np.uint8)
    cv2.circle(dmg, (400, 430), 45, MASK_FG, -1)

    add = mask_for_damage_add(img, dmg, rng=rng)
    grow = mask_for_damage_enlarge(dmg, rng=rng)
    rem = mask_for_removal(dmg, rng=rng)

    print(f"{'maske':<22} {'alan %':<10} durum")
    print("-" * 46)
    for name, m in [("mask_for_damage_add", add), ("mask_for_damage_enlarge", grow),
                    ("mask_for_removal", rem)]:
        if m is None:
            print(f"{name:<22} {'-':<10} None (uygun bolge yok)")
            continue
        print(f"{name:<22} {mask_area_frac(m)*100:<10.2f} OK")
        assert set(np.unique(m)).issubset({0, 255}), f"{name}: maske ikili degil"

    # Hasar ekleme maskesi mevcut hasarla CAKISMAMALI -- bu, M1'in tanimi.
    if add is not None:
        overlap = cv2.bitwise_and(add, dmg)
        assert overlap.sum() == 0, "M1 maskesi mevcut hasarla cakisiyor!"
        print("\nM1 maskesi mevcut hasarla cakismiyor: OK")

    # Buyutme maskesi eski hasarin ICINI icermemeli (sadece halka).
    if grow is not None:
        inner = cv2.bitwise_and(grow, erode(dmg, 4))
        assert inner.sum() == 0, "Buyutme maskesi eski hasarin icini kapsiyor!"
        print("Buyutme maskesi halka seklinde: OK")

    print("\nmasks.py sanity check OK")
