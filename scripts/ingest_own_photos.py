"""
Kendi telefon fotograflarinin ingest hatti (plan 4.2, "Kendi telefon
fotograflarin" satiri + W1 Bulgu 4).

NEDEN BU DOSYA VAR
------------------
W1 Bulgu 4: "CarDD akademik bir set, EXIF temizlenmis. L1 metadata
katmaninin anlamli olculebilmesi icin gercek EXIF'li fotograf gerekiyor
-> kendi telefon fotograflarim kritik."

Bu script o fotograflari projeye SIZINTISIZ ve KVKK-UYUMLU sokar.

UC KOPYA MANTIGI (kasitli ve onemli)
------------------------------------
    data/raw/own_photos/original/   Dokunulmamis dosya. EXIF tam.
                                    .gitignore'a girer (plaka icerebilir).
                                    L1 metadata dedektoru BUNU okur.
    data/raw/own_photos/anon/       Plaka bulaniklastirilmis, GPS silinmis,
                                    kamera EXIF'i KORUNMUS. Repoya girebilir.
                                    Gorsel katmanlar (L2/L3) BUNU kullanir.
    data/raw/own_photos/exif/       Her fotograf icin JSON yan dosya.
                                    GPS alanlari ayrilmis olarak saklanir.

Neden EXIF'i anon kopyada koruyoruz: plaka bir kisisel veridir, kamera
Make/Model degildir. EXIF'i tumden silersek L1 katmanini kendi elimizle
korlestirmis oluruz -- ki bu fotograflari cekmemizin TEK sebebi oydu.
GPS ise konum verisidir ve silinir.

ETIKETLEME
----------
Kaynak klasorde iki alt klasor bekler:
    <kaynak>/damaged/   hasarli arac fotograflari
    <kaynak>/clean/     hasarsiz arac fotograflari
Alt klasor yoksa tum dosyalar "unknown" olarak isaretlenir ve
gen_params'a yazilir (etiket yine "real"dir -- hepsi gercek fotograf).

Not: hasarli/hasarsiz ayrimi bu projede SINIF ETIKETI DEGILDIR. Ikisi de
"real"dir. Bu bilgi (a) M1 icin hangi fotografin hasarsiz panel sundugunu
secmeye, (b) "dedektor hasari mi yoksa sentetikligi mi ogreniyor" analizine
yarar (plan 4.5 Tuzak 4'un varyanti).

Calistirma:
    python scripts/ingest_own_photos.py --src "C:/Users/tunah/Pictures/arac_fotolari"
    python scripts/ingest_own_photos.py --src ./foto --no-blur-plates   # once bak, sonra karar ver
    python scripts/ingest_own_photos.py --report-only --src ./foto      # sadece EXIF raporu
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.imageio import has_non_ascii, imread, imwrite  # noqa: E402

OUT_ROOT = Path("data/raw/own_photos")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".dng"}

# Plaka tespiti icin OpenCV'nin hazir Haar siniflandiricilari.
# KABA bir arac -- kacirabilir ve yanlis pozitif verebilir. Bu yuzden
# script sonunda "gozle kontrol et" uyarisi basar ve --review-dir ile
# tespit kutularini isaretlenmis bir onizleme klasoru uretir.
CASCADES = [
    "haarcascade_russian_plate_number.xml",
    "haarcascade_license_plate_rus_16stages.xml",
]


# ---------------------------------------------------------------------------
# EXIF
# ---------------------------------------------------------------------------


def _jsonable(v):
    """EXIF degerleri IFDRational, bytes, tuple olabilir -> JSON'a cevir."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")[:200]
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if hasattr(v, "numerator") and hasattr(v, "denominator"):
        try:
            return float(v)
        except ZeroDivisionError:
            return None
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)[:200]


def read_exif(path: Path) -> tuple[dict, dict]:
    """(exif_without_gps, gps_only) doner."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return {}, {}
            base = {TAGS.get(k, str(k)): _jsonable(v) for k, v in exif.items()}

            gps = {}
            try:
                gps_ifd = exif.get_ifd(0x8825)
                gps = {GPSTAGS.get(k, str(k)): _jsonable(v) for k, v in gps_ifd.items()}
            except (AttributeError, KeyError):
                pass

            # Ic IFD'de asil kamera alanlari var (ExposureTime, ISO...)
            try:
                exif_ifd = exif.get_ifd(0x8769)
                base.update({TAGS.get(k, str(k)): _jsonable(v) for k, v in exif_ifd.items()})
            except (AttributeError, KeyError):
                pass

            base.pop("GPSInfo", None)
            return base, gps
    except Exception as e:
        return {"_error": str(e)}, {}


def exif_quality(exif: dict) -> str:
    """Fotografin L1 icin ne kadar degerli oldugunu ozetler."""
    if not exif or "_error" in exif:
        return "YOK"
    has_cam = bool(exif.get("Make") or exif.get("Model"))
    has_time = bool(exif.get("DateTimeOriginal") or exif.get("DateTime"))
    if has_cam and has_time:
        return "TAM"
    if has_cam or has_time:
        return "KISMI"
    return "ZAYIF"


# ---------------------------------------------------------------------------
# Plaka bulaniklastirma (KVKK)
# ---------------------------------------------------------------------------


def _load_cascades() -> list:
    """Haar siniflandiricilarini yukler; yuklenemezse BOS LISTE doner.

    OpenCV 5.0 ILE GELEN KIRILMA
    ----------------------------
    OpenCV 5.0'da `cv2.CascadeClassifier` TAMAMEN KALDIRILDI. Dogrulama:

        opencv-python-headless==5.0.0.93
        hasattr(cv2, "CascadeClassifier")  -> False
        [n for n in dir(cv2) if "ascade" in n]  -> []

    requirements.txt surumu sabitlemedigi icin taze kurulumlar 5.x cekiyor.
    Bu fonksiyon artik her iki surumde de CALISIR: cascade yoksa plaka
    bulaniklastirma sessizce KAPANIR, ingest'in geri kalani (asil is olan
    EXIF cikarma ve GPS ayirma) etkilenmez.

    Bilincli tasarim karari: anonimlestirme YAPILAMADIGINDA program cokmez
    ama KULLANICIYI ACIKCA UYARIR (bkz. main() sonundaki uyarilar). Sessizce
    "anonimlestirdim" demek, cokmekten cok daha tehlikeli olurdu.
    """
    if not hasattr(cv2, "CascadeClassifier"):
        return []
    haar_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if not haar_dir:
        return []

    out = []
    for name in CASCADES:
        try:
            c = cv2.CascadeClassifier(haar_dir + name)
        except cv2.error:
            continue
        if not c.empty():
            out.append(c)
    return out


def cascade_unavailable_reason() -> str | None:
    """Cascade yoksa sebebini insan diliyle anlatir (None = sorun yok)."""
    if not hasattr(cv2, "CascadeClassifier"):
        return (
            f"OpenCV {cv2.__version__} surumunde cv2.CascadeClassifier YOK "
            f"(5.0 ile kaldirildi). Plaka bulaniklastirma bu surumde "
            f"calisamaz."
        )
    if not getattr(getattr(cv2, "data", None), "haarcascades", None):
        return f"OpenCV {cv2.__version__}: cv2.data.haarcascades yolu bulunamadi."
    if not _load_cascades():
        return f"OpenCV {cv2.__version__}: Haar xml dosyalari yuklenemedi."
    return None


def detect_plates(img_bgr: np.ndarray, cascades: list) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    boxes: list[tuple[int, int, int, int]] = []
    for c in cascades:
        for (x, y, w, h) in c.detectMultiScale(gray, 1.08, 4, minSize=(60, 18)):
            boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def blur_boxes(img_bgr: np.ndarray, boxes, pad: float = 0.18) -> np.ndarray:
    """Kutulari guclu Gauss + pikselleme ile kapatir.

    Sadece blur YETMEZ: yeterince guclu olmayan bir blur ters cevrilebilir
    ve karakterler okunabilir kalir. Once pikselleme (bilgi yok edici),
    sonra blur (kenar yumusatma) uygulaniyor."""
    out = img_bgr.copy()
    H, W = out.shape[:2]
    for (x, y, w, h) in boxes:
        px, py = int(w * pad), int(h * pad)
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(W, x + w + px), min(H, y + h + py)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = out[y0:y1, x0:x1]
        small = cv2.resize(roi, (max(1, (x1 - x0) // 12), max(1, (y1 - y0) // 8)),
                           interpolation=cv2.INTER_LINEAR)
        roi = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        k = max(9, ((min(x1 - x0, y1 - y0) // 4) * 2) + 1)
        out[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


ORIENTATION_TAG = 274  # EXIF 0th IFD: Orientation


def save_with_exif(
    img_bgr: np.ndarray, dst: Path, src: Path, *, strip_gps: bool = True
) -> int:
    """Anon kopyayi kaydeder ve ORIJINAL EXIF'i (GPS haric) tasir.

    piexif kullaniyoruz cunku PIL, exif'i yeniden yazarken bazi alanlari
    dusuruyor; L1 dedektorunun okudugu Make/Model/Software alanlarinin
    birebir korunmasi gerekiyor.

    YON (ORIENTATION) NORMALIZASYONU -- KRITIK
    ------------------------------------------
    iPhone fotografi YATAY saklayip EXIF'e "beni 90 derece dondur"
    (Orientation=6) etiketi koyar. Iki kutuphane bu etikete FARKLI
    davranir -- olculdu:

        kaynak dosya            : 4032x3024 saklanmis, Orientation=6
        PIL   Image.open().size : (4032, 3024)   <- etiketi UYGULAMAZ
        cv2   imdecode().shape  : (4032, 3024)?  HAYIR -> (3024, 4032)
                                                  <- etiketi UYGULAR

    Bu fark sessiz bir felakettir: maskeler cv2 ile uretilir, laundering
    PIL ile calisir. Ayni dosya icin biri dikey biri yatay gorurse
    MASKE ILE GORUNTU BIRBIRINI TUTMAZ ve piksel-F1 anlamsizlasir.
    Hata mesaji da vermez -- sadece yanlis sayilar uretir.

    Cozum: gelen `img_bgr` zaten cv2 ile okunmus, yani PIKSELLER ARTIK
    GORUNTULENME YONUNDE. O halde EXIF'teki Orientation degeri de 1
    ("dondurme") yapilmali. Aksi halde goruntuleyici bir kez daha dondurur
    ve fotograf yan yatar.

    Doner: orijinal Orientation degeri (1 veya yoksa 0) -- raporlama icin.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    exif_bytes = None
    orig_orientation = 0
    if src.suffix.lower() in {".jpg", ".jpeg"}:
        try:
            import piexif

            d = piexif.load(str(src))
            orig_orientation = int(d["0th"].get(ORIENTATION_TAG, 1))
            if strip_gps:
                d["GPS"] = {}
            d.pop("thumbnail", None)  # thumbnail plakayi tasiyabilir!
            d["1st"] = {}
            # Pikseller zaten dondurulmus durumda -> etiket sifirlanir.
            d["0th"][ORIENTATION_TAG] = 1
            exif_bytes = piexif.dump(d)
        except Exception:
            exif_bytes = None

    if exif_bytes:
        pil.save(dst, "JPEG", quality=95, subsampling=2, exif=exif_bytes)
    else:
        pil.save(dst, "JPEG", quality=95, subsampling=2)

    return orig_orientation


# ---------------------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------------------


def label_from_path(p: Path, src_root: Path) -> str:
    parts = {q.lower() for q in p.relative_to(src_root).parts[:-1]}
    if {"damaged", "hasarli", "hasarli_arac"} & parts:
        return "damaged"
    if {"clean", "temiz", "hasarsiz", "normal"} & parts:
        return "clean"
    return "unknown"


def short_id(p: Path) -> str:
    return hashlib.md5(p.name.encode()).hexdigest()[:8]


def main() -> None:
    ap = argparse.ArgumentParser(description="Kendi fotograflarini projeye al")
    ap.add_argument("--src", required=True, help="Fotograflarin bulundugu klasor")
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--no-blur-plates", action="store_true",
                    help="Plaka bulaniklastirmayi atla (KVKK riski -- bilincli kullan)")
    ap.add_argument("--report-only", action="store_true",
                    help="Hicbir sey kopyalama, sadece EXIF raporu bas")
    ap.add_argument("--review-dir", default="",
                    help="Tespit kutulari cizilmis onizlemeleri buraya yaz")
    a = ap.parse_args()

    src_root = Path(a.src)
    if not src_root.exists():
        print(f"HATA: kaynak klasor yok: {src_root}")
        sys.exit(1)

    files = sorted(p for p in src_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"HATA: {src_root} altinda goruntu bulunamadi.")
        sys.exit(1)

    print(f"{len(files)} dosya bulundu: {src_root}\n")

    out_root = Path(a.out)
    dir_orig = out_root / "original"
    dir_anon = out_root / "anon"
    dir_exif = out_root / "exif"
    review = Path(a.review_dir) if a.review_dir else None

    # Klasorleri BASTAN olustur. Onceden dongu icinde olusturuluyordu; tek
    # bir goruntu bile okunamayinca hicbiri olusmuyor ve ozet dosyasi
    # FileNotFoundError ile patliyordu. Cikti dizini, girdinin okunup
    # okunmamasindan bagimsiz olarak var olmali.
    if not a.report_only:
        for d in (out_root, dir_orig, dir_anon, dir_exif):
            d.mkdir(parents=True, exist_ok=True)

    # Rapor modunda hicbir dosya yazilmadigi icin cascade'e de gerek yok.
    cascades: list = []
    plate_blur_off_reason: str | None = None
    if a.report_only:
        plate_blur_off_reason = "rapor modu (--report-only): dosya yazilmiyor"
    elif a.no_blur_plates:
        plate_blur_off_reason = "--no-blur-plates ile elle kapatildi"
    else:
        cascades = _load_cascades()
        if not cascades:
            plate_blur_off_reason = cascade_unavailable_reason()

    if plate_blur_off_reason:
        print(f"NOT: Plaka bulaniklastirma KAPALI -- {plate_blur_off_reason}\n")

    qual_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    make_counter: Counter[str] = Counter()
    rows = []
    total_boxes = 0
    no_plate_found = 0
    unreadable = 0
    rotated = 0

    print(f"{'dosya':<30}{'etiket':<10}{'EXIF':<8}{'kamera':<24}{'plaka'}")
    print("-" * 84)

    for p in files:
        exif, gps = read_exif(p)
        q = exif_quality(exif)
        lab = label_from_path(p, src_root)
        cam = f"{exif.get('Make','')} {exif.get('Model','')}".strip() or "-"
        qual_counter[q] += 1
        label_counter[lab] += 1
        make_counter[cam] += 1

        n_boxes = 0
        if not a.report_only:
            sid = f"own_{lab}_{short_id(p)}"
            img = imread(p)
            if img is None:
                unreadable += 1
                print(f"{p.name[:29]:<30}{lab:<10}{q:<8}{cam[:23]:<24}OKUNAMADI")
                continue

            # 1) Orijinali dokunmadan sakla (EXIF tam, L1 bunu okuyacak)
            shutil.copy2(p, dir_orig / f"{sid}{p.suffix.lower()}")

            # 2) Anonimlestirilmis kopya
            boxes = detect_plates(img, cascades) if cascades else []
            n_boxes = len(boxes)
            total_boxes += n_boxes
            if n_boxes == 0 and cascades:
                no_plate_found += 1
            anon = blur_boxes(img, boxes) if boxes else img
            orient = save_with_exif(anon, dir_anon / f"{sid}.jpg", p)
            if orient not in (0, 1):
                rotated += 1

            # Onizleme SADECE cascade calisiyorsa uretilir. Cascade yokken
            # uretmek, her fotografin ~4 MB'lik ikinci bir kopyasini diske
            # yazmak demektir ve uzerinde isaretlenecek hicbir kutu yoktur.
            if review is not None and cascades:
                review.mkdir(parents=True, exist_ok=True)
                vis = img.copy()
                for (x, y, w, h) in boxes:
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
                imwrite(review / f"{sid}_review.jpg", vis)

            # 3) EXIF yan dosyasi (GPS ayri tutulur)
            with open(dir_exif / f"{sid}.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source_image_id": sid,
                        "original_filename": p.name,
                        "damage_label": lab,
                        "exif_quality": q,
                        "exif": exif,
                        "gps_removed_from_anon": gps,
                        "plate_boxes_blurred": n_boxes,
                        "exif_orientation_normalized": True,
                    },
                    f, ensure_ascii=False, indent=2,
                )
            rows.append({"source_image_id": sid, "damage_label": lab, "exif_quality": q})

        print(f"{p.name[:29]:<30}{lab:<10}{q:<8}{cam[:23]:<24}"
              f"{n_boxes if not a.report_only else '-'}")

    print("\n" + "=" * 62)
    print("OZET")
    print("=" * 62)
    print(f"Toplam dosya        : {len(files)}")
    if not a.report_only:
        print(f"Basariyla islenen   : {len(rows)}")
        print(f"Okunamayan          : {unreadable}")
        print(f"Yonu normalize edilen: {rotated}  (EXIF Orientation -> 1)")
    print(f"EXIF kalitesi       : {dict(qual_counter)}")
    print(f"Hasar etiketi       : {dict(label_counter)}")
    print(f"Kameralar           : {dict(make_counter)}")

    if not a.report_only:
        if cascades:
            print(f"Bulaniklastirilan plaka kutusu : {total_boxes}")
            print(f"Hic plaka bulunamayan foto     : {no_plate_found}/{len(files)}")
        else:
            print(f"Plaka bulaniklastirma          : YAPILMADI ({plate_blur_off_reason})")
        with open(out_root / "ingest_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {"n_files": len(files), "exif_quality": dict(qual_counter),
                 "damage_labels": dict(label_counter), "cameras": dict(make_counter),
                 "plate_blurring_applied": bool(cascades),
                 "orientation_normalized": rotated,
                 "plate_blur_off_reason": plate_blur_off_reason,
                 "plate_boxes": total_boxes, "rows": rows},
                f, ensure_ascii=False, indent=2,
            )
        print(f"\nYazildi:\n  {dir_orig}/  (GITIGNORE -- plaka icerebilir)")
        if cascades:
            print(f"  {dir_anon}/  (plakalar bulaniklastirildi)")
        else:
            print(f"  {dir_anon}/  (DIKKAT: plakalar BULANIKLASTIRILMADI)")
        print(f"  {dir_exif}/  (GPS burada, GITIGNORE)")

    # --- Uyarilar ---
    if qual_counter["YOK"] + qual_counter["ZAYIF"] > len(files) * 0.3:
        print("\n  >>> UYARI: Fotograflarin buyuk kismi EXIF'siz.")
        print("  >>> Muhtemelen WhatsApp/Telegram uzerinden aktarilmis.")
        print("  >>> Kablo veya Google Drive ile yeniden aktar -- W1 Bulgu 4'e")
        print("  >>> gore L1 katmanini dogrulamanin TEK yolu bu fotograflar.")

    if unreadable:
        print(f"\n  >>> UYARI: {unreadable} dosya okunamadi.")
        if has_non_ascii(src_root) or has_non_ascii(Path.cwd()):
            print("  >>> Yolda ASCII disi karakter var (orn. 'Masaustu' icindeki 'u').")
            print("  >>> src/data/imageio.py bunu cozmus olmali; hala olmuyorsa")
            print("  >>> dosyalari ASCII bir yola tasi (orn. C:/veri/arac_fotolari).")
        else:
            print("  >>> Dosyalar bozuk veya desteklenmeyen formatta olabilir.")

    if label_counter["unknown"] == len(files):
        print("\n  >>> NOT: Hasarli/hasarsiz ayrimi yapilamadi.")
        print("  >>> Kaynak klasorde 'damaged/' ve 'clean/' alt klasorleri olustur.")

    if not a.report_only and cascades and total_boxes == 0:
        print("\n  >>> UYARI: Hicbir fotografta plaka tespit edilemedi.")
        print("  >>> Haar siniflandiricisi kabadir. --review-dir ile onizleme")
        print("  >>> uretip GOZLE kontrol et; gerekirse elle bulaniklastir.")

    if not a.report_only and not cascades:
        print("\n  >>> KVKK NOTU: Plakalar bulaniklastirilmadi.")
        print("  >>> Bu ARASTIRMA AKISINI engellemez: data/raw/ tamamen")
        print("  >>> .gitignore'da, yani plakali goruntuler GitHub'a cikmaz.")
        print("  >>> ANCAK su uc durumdan once mutlaka anonimlestir:")
        print("  >>>   1. veri setini biriyle paylasmadan once")
        print("  >>>   2. sunuma/dokumana ornek gorsel koymadan once")
        print("  >>>   3. dataset_card.md icin kolaj uretmeden once")

    print("\nSonraki adim: python scripts/build_manifest_v2.py")


if __name__ == "__main__":
    main()
