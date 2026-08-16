"""
Unicode-guvenli goruntu okuma/yazma.

NEDEN BU MODUL VAR
------------------
`cv2.imread` / `cv2.imwrite`, Windows'ta dosya yolunu dar karakterli (ANSI)
API'ye gecirir. Yol ASCII disi bir karakter iceriyorsa dosya BULUNAMAZ ve
fonksiyon sessizce None doner ya da False ile basarisiz olur.

Bu projede yol SURKLI ASCII disi:

    C:\\Users\\tunah\\OneDrive\\Masaustu\\...      <- "Masaüstü" icindeki 'ü'

Gercek hata ciktisi:

    cv::findDecoder imread_('...Masa├╝st├╝\\arac_fotolari\\clean\\IMG_3848.jpeg'):
        can't open/read file: check file path/integrity

Dikkat: dosya BOZUK DEGIL. PIL ayni dosyayi sorunsuz aciyor (EXIF okuma
calisti). Sorun tamamen cv2'nin yol kodlamasinda.

COZUM
-----
Dosyayi Python ile (unicode-guvenli) byte olarak oku, cv2'ye BELLEKTEN ver:

    np.fromfile(path) -> cv2.imdecode(...)
    cv2.imencode(...) -> tofile(path)

Bu, platformdan bagimsiz calisir ve Linux/Colab'da da dogrudur.

KURAL: Bu projede cv2.imread/cv2.imwrite DOGRUDAN CAGRILMAZ. Her zaman
buradaki imread/imwrite kullanilir.


IKINCI TUZAK: EXIF ORIENTATION (cv2 ile PIL AYNI SEYI YAPMAZ)
--------------------------------------------------------------
Telefon fotograflari sahneyi YATAY saklayip EXIF'e "beni dondur"
(Orientation=6) etiketi koyar. Olculen davranis, ayni dosya icin:

    saklanan boyut          : 4032x3024, Orientation=6
    PIL   Image.open().size : (4032, 3024)   <- etiketi UYGULAMAZ
    cv2   imdecode().shape  : (3024, 4032)   <- etiketi UYGULAR

Yani bu modulun imread'i ile PIL, ayni dosyayi FARKLI YONDE dondurur.

Bu neden tehlikeli: maskeler cv2 ile uretiliyor (src/data/masks.py),
laundering PIL ile calisiyor (src/data/launder.py). Yon etiketi olan bir
goruntude maske ile goruntu birbirini tutmaz -- ve hata mesaji cikmaz,
sadece piksel-F1 sessizce coker.

Bu yuzden scripts/ingest_own_photos.py, kendi telefon fotograflarini
projeye alirken pikselleri goruntulenme yonune sabitler ve EXIF
Orientation degerini 1 yapar. Yani manifest'e giren HER goruntude
cv2 ile PIL ayni seyi gorur.

Disaridan yeni bir kaynak eklerken (Roboflow, meslektas fotograflari)
ayni normalizasyondan gecirmeyi unutma; `imread_ignore_orientation`
karsilastirma icin buradadir.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """cv2.imread'in unicode-guvenli karsiligi. Okunamazsa None doner."""
    path = Path(path)
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imread_gray(path: str | Path) -> np.ndarray | None:
    return imread(path, cv2.IMREAD_GRAYSCALE)


def imread_ignore_orientation(path: str | Path) -> np.ndarray | None:
    """EXIF Orientation etiketini UYGULAMADAN okur -- yani PIL ile ayni
    sonucu verir. Teshis amaclidir: bir goruntude yon etiketi sorunu olup
    olmadigini `imread` ciktisiyla karsilastirarak anlarsin."""
    return imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


def orientation_mismatch(path: str | Path) -> bool:
    """cv2 ile PIL bu dosyayi farkli yonde mi goruyor?

    True donerse goruntu normalize edilmemis demektir ve maske/goruntu
    uyusmazligi riski vardir (bkz. modul basligi)."""
    a = imread(path)
    b = imread_ignore_orientation(path)
    if a is None or b is None:
        return False
    return a.shape[:2] != b.shape[:2]


def imwrite(path: str | Path, img: np.ndarray, params: list[int] | None = None) -> bool:
    """cv2.imwrite'in unicode-guvenli karsiligi.

    Uzantiya gore encode eder; ust klasoru gerekirse olusturur.
    Basarisizsa False doner (sessizce yutmaz -- cagiran kontrol etmeli).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    return True


def has_non_ascii(path: str | Path) -> bool:
    """Yol ASCII disi karakter iceriyor mu? Teshis mesajlari icin."""
    return not str(path).isascii()


if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (120, 200, 3), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as td:
        # Kasitli olarak Turkce karakterli bir yol kuruyoruz -- bu testin
        # tamami zaten bunun icin var.
        tricky = Path(td) / "Masaüstü" / "araç_fotoğrafları"
        tricky.mkdir(parents=True)
        p = tricky / "örnek_görüntü.jpg"

        print(f"Test yolu ASCII disi mi: {has_non_ascii(p)}")

        # 1) Bizim imwrite/imread
        assert imwrite(p, img), "imwrite basarisiz"
        back = imread(p)
        assert back is not None, "imread None dondu"
        assert back.shape == img.shape, f"sekil uyusmuyor: {back.shape} != {img.shape}"
        print(f"imwrite + imread      : OK  {back.shape}")

        # 2) Ham cv2 ayni yolda ne yapiyor? (Windows'ta None doner,
        #    Linux'ta calisir -- ikisi de bilgilendirici)
        raw = cv2.imread(str(p))
        print(f"ham cv2.imread        : {'None (beklenen: Windows)' if raw is None else 'calisti (Linux/macOS)'}")

        # 3) Gri okuma
        g = imread_gray(p)
        assert g is not None and g.ndim == 2
        print(f"imread_gray           : OK  {g.shape}")

        # 4) PNG (maskeler icin) -- kayipsiz olmali
        pm = tricky / "maske_çıktısı.png"
        mask = (rng.integers(0, 2, (120, 200), dtype=np.uint8)) * 255
        assert imwrite(pm, mask)
        mb = imread_gray(pm)
        assert np.array_equal(mask, mb), "PNG round-trip kayipsiz degil!"
        print("PNG kayipsiz round-trip: OK")

    print("\nimageio.py sanity check OK")
