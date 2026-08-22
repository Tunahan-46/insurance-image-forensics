"""
M3 katmani -- klasik (AI'siz) manipulasyon (plan 4.2, tablo M3).

NEDEN ONEMLI: Dolandiricinin elinde her zaman bir difuzyon modeli yok.
Telefonundaki "kopyala-yapistir" ya da basit bir foto editoru ile yapilan
manipulasyonlar hala en yaygin olanlardir. Ayrica bu ornekler AI izi
TASIMAZ -- yani "difuzyon artefakti ariyorum" diyen bir dedektor bunlarda
tamamen basarisiz olur. Bu, Hafta 5 fuzyon katmaninin varlik sebebidir.

Uc alt tip:
    copy_move  : AYNI goruntuden bir bolge kopyalanip baska yere yapistirilir
                 (orn. bir cizigi ikinci kez gostermek)
    splice     : BASKA bir goruntunun hasar bolgesi kesilip yapistirilir
    bg_replace : arac korunur, arka plan baska bir sahneyle degistirilir
                 (olay yerini uydurma)

Hepsi GPU'suz, saniyeler icinde calisir. Maskeler zemin gercegidir.

W3 DUZELTMESI -- GOZLE DENETIMDE BULUNAN IKI SORUN
---------------------------------------------------
dataset_card.md icin kolaj uretilirken (scripts/make_collage.py) iki ayri
kok neden bulundu, ikisi de bu dosyada duzeltildi:

  1. splice/copy_move'da renk uyumsuzlugu. `_paste_region` %70/%60
     olasilikla cv2.seamlessClone kullaniyordu ama Poisson blending
     BUYUK/dokulu yamalarda merkez rengini degistirmez (sadece sinirdan
     iceri sizar). Olcum (scripts/diag_splice.py): splice orneklerinin
     %31'i "alakasiz renk" esiginin (dE>25) uzerindeydi. Duzeltme:
     harmanlamadan ONCE `masks.color_transfer` ile yamanin rengi hedefin
     yerel rengine yaklastiriliyor; harmanlama SONRASI da
     `masks.color_consistency_de` ile ikinci bir kabul kapisi eklendi.

  2. bg_replace'te dikdortgen maske. `vehicle_region` (GrabCut) yakin
     cekim goruntulerde anlamli bir on/arka plan ayrimi bulamayip ROI
     dikdortgenine yakin bir sonuc uretebiliyordu (cardd_003639_bg_replace
     -- dumduz kirmizi bir dikdortgen dogrudan farin ustune yapismisti).
     Bu HEM gercekci degil HEM DE plan 4.5'in acikca uyardigi "dikdortgen
     maske -> ogrenilebilir kestirme yol" tuzagi. Duzeltme:
     `masks.shape_is_rectangular` ile bu durum tespit edilip reddediliyor.

Calistirma:
    python -m src.data.generators.classic_manip --manifest data/processed/manifest_v1.parquet --n 400
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.data.generators import GenResult
from src.data.imageio import imread, imwrite
from src.data.masks import (
    changed_fraction_in_mask,
    color_consistency_de,
    color_transfer,
    dilate,
    leak_fraction_outside_mask,
    load_mask,
    mask_area_frac,
    random_blob,
    roughen,
    sample_point_in,
    save_mask,
    shape_is_rectangular,
    vehicle_region,
)

DEFAULT_OUT = Path("data/raw/manipulated/classic")
SUBTYPES = ("copy_move", "splice", "bg_replace")

# KABUL KAPISI (bkz. masks.changed_fraction_in_mask):
# Maske icindeki piksellerin en az %25'i gercekten degismis olmali.
# Altinda kalan ornek DISKE YAZILMAZ. Gerekce: "manipule" etiketli ama
# orijinalle ayni bir goruntu, hem goruntu-seviyesi etiketi hem de
# piksel zemin gercegini yalanlar.
MIN_CHANGED_IN_MASK = 0.25

# IKINCI KABUL KAPISI -- W3 duzeltmesi (bkz. dosya basi docstring).
# color_transfer harmanlamadan once rengi hedefe yaklastirir ama garanti
# etmez (kucuk/parcali maskede olcum gurultulu olabilir, ya da seamless
# sinir pikselinde renk transferini kismen ezebilir). Bu, SONUCU olcen
# bagimsiz bir dogrulama: maske ici hala cevresinden "alakasiz" derecede
# farkliysa (bkz. masks.color_consistency_de docstring, dE>25 esigi)
# diske YAZILMAZ. Sadece splice/copy_move icin uygulanir -- bg_replace
# zaten farkli bir sahne yapistiriyor, yuksek dE orada beklenen ve normal.
MAX_COLOR_DE = 25.0

# bg_replace icin ton harmanlama gucu -- W3 2. tur duzeltme (bkz.
# bg_replace docstring). 1.0 = donor arka plan tamamen orijinal sahnenin
# renk istatistiklerine cekilir (farkli sahne hissi kaybolur); 0.0 = hic
# dokunulmaz (eski davranis). 0.5, "farkli ama uyumlu ton" dengesini
# hedefliyor.
BG_TONE_MATCH_STRENGTH = 0.5

# bg_replace icin UCUNCU kabul kapisi: renk-transferi + genis harmanlamadan
# SONRA bile bazi ornekler asiri kaliyor (donor sahnenin ton/parlakligi
# orijinalden cok uzaksa color_transfer tek basina yetmeyebilir). Esik,
# splice/copy_move'daki MAX_COLOR_DE'den kasten FARKLI ve DAHA GEVSEK:
# bg_replace maskesinin dogal (manipulasyonsuz) taban dE'si zaten yuksek
# (~16-40, cunku arac kenari ile arka plan gercek fotograflarda da farkli
# renktedir -- bkz. diag_splice.py "dE kaynak" sutunu). Bu yuzden MUTLAK
# dE yerine "artis" (manip dE - kaynak dE) olculur: manipulasyonun KENDISI
# ne kadar EK renk kopuklugu getirdi. 30'un uzerindeki artis, dogal sahne
# farkiyla aciklanamayacak kadar buyuk kabul edildi (olculen en kotu 10
# ornekte artis 36-89 araligindaydi).
BG_MAX_COLOR_ARTIS = 30.0


def _paste_region(
    dst_bgr: np.ndarray,
    src_patch: np.ndarray,
    mask: np.ndarray,
    center: tuple[int, int],
    *,
    seamless: bool = True,
    color_match: bool = True,
) -> np.ndarray:
    """Maskeli bolgeyi hedefe yapistirir.

    seamless=True -> cv2.seamlessClone (Poisson blending). Renk/isik
    uyumsuzlugunu giderdigi icin gozle tespiti zorlastirir; buna karsilik
    gradyan alaninda kendine ozgu bir iz birakir. Bu ikilik tam olarak
    dedektorun ogrenmesini istedigimiz seydir.

    seamless=False -> duz alfa harmanlama. Daha 'amator' bir saldiri;
    veri setinde ikisinin de bulunmasi zorluk cesitliligi saglar.

    color_match=True (varsayilan) -> harmanlamadan ONCE `src_patch`'in
    maske icindeki rengi, dst_bgr'nin maskeyi cevreleyen halkasina Lab
    uzayinda yaklastirilir (bkz. masks.color_transfer). W3'te gozle
    bulundu: buyuk/dokulu yamalarda Poisson blending TEK BASINA merkez
    rengini degistirmiyor (sadece sinirdan sizar), duz alfa ise HIC
    degistirmiyor -- ikisi de iki farkli aracin renginin yamada oldugu
    gibi kalmasina yol aciyordu.
    """
    cy, cx = center
    if color_match:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (53, 53))
        k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        binm = (mask > 127).astype(np.uint8)
        ring = (cv2.dilate(binm, k) - cv2.dilate(binm, k_in)).clip(0, 1) * 255
        src_patch = color_transfer(src_patch, mask, dst_bgr, ring.astype(np.uint8))
    if seamless:
        try:
            return cv2.seamlessClone(
                src_patch, dst_bgr, (mask > 127).astype(np.uint8) * 255,
                (cx, cy), cv2.NORMAL_CLONE,
            )
        except cv2.error:
            pass  # merkez kenara cok yakinsa OpenCV atar -> alfa'ya dus
    a = (cv2.GaussianBlur(mask, (15, 15), 0).astype(np.float32) / 255.0)[..., None]
    return (src_patch.astype(np.float32) * a + dst_bgr.astype(np.float32) * (1 - a)).astype(np.uint8)


def _shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = mask.shape[:2]
    return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


def _shift_image(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def copy_move(
    img_bgr: np.ndarray, damage_mask: np.ndarray | None, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray] | None:
    """Ayni goruntuden bolge kopyala-yapistir.

    Hasar maskesi varsa ONU kopyalariz (gercekci: var olan cizigi ikinci
    kez gostermek). Yoksa arac uzerinden rastgele bir blob.

    Zemin gercegi = YAPISTIRILAN bolge (kaynak bolge degil; kaynak orijinaldir).
    """
    h, w = img_bgr.shape[:2]
    # damage_mask (CarDD'nin bilinen hasar konumu) GrabCut'a tohum olarak
    # veriliyor -- W3 3. tur duzeltmesi, bkz. masks.vehicle_region docstring.
    body = vehicle_region(img_bgr, hint_mask=damage_mask)

    if damage_mask is not None and mask_area_frac(damage_mask) > 0.002:
        src_mask = dilate(damage_mask, 6)
    else:
        pt = sample_point_in(body, rng)
        if pt is None:
            return None
        src_mask = roughen(random_blob(h, w, rng=rng, center=pt, radius_frac=0.09), rng)

    if mask_area_frac(src_mask) < 0.001:
        return None

    ys, xs = np.nonzero(src_mask > 127)
    src_c = (int(ys.mean()), int(xs.mean()))

    # Hedef: aracin uzerinde, kaynaktan yeterince uzak bir nokta
    for _ in range(30):
        dst_c = sample_point_in(body, rng)
        if dst_c is None:
            return None
        dy, dx = dst_c[0] - src_c[0], dst_c[1] - src_c[1]
        if abs(dy) + abs(dx) < 0.15 * (h + w):
            continue  # cok yakin: fark edilmez bir 'manipulasyon' olur
        moved_mask = _shift_mask(src_mask, dy, dx)
        # Kaydirilmis maske goruntu disina tastiysa ise yaramaz
        if mask_area_frac(moved_mask) < 0.8 * mask_area_frac(src_mask):
            continue
        moved_img = _shift_image(img_bgr, dy, dx)
        out = _paste_region(img_bgr, moved_img, moved_mask, dst_c, seamless=rng.random() < 0.6)
        return out, moved_mask
    return None


def splice(
    img_bgr: np.ndarray,
    donor_bgr: np.ndarray,
    donor_mask: np.ndarray | None,
    rng: np.random.Generator,
    *,
    dst_damage_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Baska bir aracin hasar bolgesini kes, bu goruntuye yapistir.
    Plan tablo 4.1 #3: 'baska bir aracin hasar fotografini kullanma'.

    dst_damage_mask: HEDEF (img_bgr) goruntunun kendi CarDD hasar konumu --
    varsa `vehicle_region`'a GrabCut tohumu olarak veriliyor (W3 3. tur
    duzeltmesi, bkz. masks.vehicle_region docstring). donor_mask zaten
    ayni amacla donor tarafinda kullaniliyor (yamanin KENDISI olarak)."""
    h, w = img_bgr.shape[:2]
    donor_bgr = cv2.resize(donor_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    if donor_mask is not None and mask_area_frac(donor_mask) > 0.002:
        dm = cv2.resize(donor_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        patch_mask = roughen(dilate(dm, 8), rng)
    else:
        pt = sample_point_in(vehicle_region(donor_bgr), rng)
        if pt is None:
            return None
        patch_mask = roughen(random_blob(h, w, rng=rng, center=pt, radius_frac=0.10), rng)

    body = vehicle_region(img_bgr, hint_mask=dst_damage_mask)
    for _ in range(30):
        dst_c = sample_point_in(body, rng)
        if dst_c is None:
            return None
        ys, xs = np.nonzero(patch_mask > 127)
        if len(ys) == 0:
            return None
        src_c = (int(ys.mean()), int(xs.mean()))
        dy, dx = dst_c[0] - src_c[0], dst_c[1] - src_c[1]
        moved_mask = _shift_mask(patch_mask, dy, dx)
        if mask_area_frac(moved_mask) < 0.8 * mask_area_frac(patch_mask):
            continue
        moved_donor = _shift_image(donor_bgr, dy, dx)
        out = _paste_region(img_bgr, moved_donor, moved_mask, dst_c, seamless=rng.random() < 0.7)
        return out, moved_mask
    return None


def bg_replace(
    img_bgr: np.ndarray,
    bg_bgr: np.ndarray,
    rng: np.random.Generator,
    *,
    damage_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Araci koru, arka plani degistir (olay yerini uydurma).

    Zemin gercegi maskesi = ARKA PLAN (aracin TERSI). Bu, veri setindeki
    diger senaryolarin tersi bir topolojidir: manipule alan buyuk ve
    baglantili. Localization modelinin ikisini de gormesi onemli, aksi
    halde 'kucuk leke ara' onyargisi ogrenir.

    W3 DUZELTMESI (2. tur): dikdortgen-maske kapisi (shape_is_rectangular)
    en kotu ornekleri aciklamadi -- `scripts/diag_bg_shape.py` ile olculdu,
    extent ile dE arasinda korelasyon YOK (en yuksek 'artis'li 20 ornegin
    ortalama extent'i 0.678, genelin ortalamasi 0.675 -- ayni). Yani kok
    neden segmentasyon SEKLI degil: donor arka planin genel ton/parlakligi
    orijinal sahneden cok farkli olabiliyor ve sabit 21x21 Gauss harmanlama
    bunu gizlemeye yetmiyor (o sadece GEOMETRIK gecisi yumusatir, RENK/ISIK
    uyusmazligini degil). Duzeltme: (1) donor arka plan, orijinal sahnenin
    KENDI arka planinin renk istatistiklerine `masks.color_transfer` ile
    KISMEN (strength=0.5 -- tam degil, hala 'farkli sahne' kalsin diye)
    yaklastiriliyor; (2) harmanlama cekirdegi goruntu boyutuyla olceklendi.
    Kalan asiri uc ornekler `generate()` icinde ayrica bir dE kapisindan
    (BG_MAX_COLOR_ARTIS) geciyor.

    W3 DUZELTMESI (3. tur): yeniden uretilen kolajda GOZLE bulundu --
    GrabCut istisna atmadan yakinsiyor ama goruntunun kabaca yarisini duz
    olmayan (dolayisiyla shape_is_rectangular tarafindan YAKALANMAYAN) bir
    sinirla "arac" saniyor. `damage_mask` -- CarDD'nin kendi hasar konumu,
    aracin KESIN uzerinde -- verilirse GrabCut'a tohum olarak geciliyor
    (bkz. masks.vehicle_region docstring); bu, dogru nesneye kilitlenme
    olasiligini buyuk olcude artiriyor.
    """
    h, w = img_bgr.shape[:2]
    body = vehicle_region(img_bgr, hint_mask=damage_mask)
    frac = mask_area_frac(body)
    if not (0.10 < frac < 0.85):
        return None  # segmentasyon guvenilmez
    if shape_is_rectangular(body):
        # GrabCut yakinsamadi ve vehicle_region dikdortgen fallback'e
        # dustu (bkz. masks.vehicle_region). Plan bunu acikca yasakliyor;
        # bu ornegi uretmek yerine reddediyoruz (W3 bulgusu).
        return None

    bg_mask = cv2.bitwise_not(body)
    bg = cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
    full_mask = np.full((h, w), 255, dtype=np.uint8)
    bg = color_transfer(bg, full_mask, img_bgr, bg_mask, strength=BG_TONE_MATCH_STRENGTH)

    # Harmanlama genisligi resme oranli: sabit 21px, buyuk/yuksek
    # cozunurluklu fotograflarda geciyi sert birakiyordu.
    k = max(21, (min(h, w) // 20) | 1)  # tek sayi olmali
    a = (cv2.GaussianBlur(bg_mask, (k, k), 0).astype(np.float32) / 255.0)[..., None]
    out = (bg.astype(np.float32) * a + img_bgr.astype(np.float32) * (1 - a)).astype(np.uint8)
    return out, bg_mask


def generate(
    manifest_path: str | Path,
    n: int,
    *,
    out_root: str | Path = DEFAULT_OUT,
    seed: int = 0,
    weights: dict[str, float] | None = None,
    splits: list[str] | None = None,
    resume: bool = True,
) -> list[GenResult]:
    """n adet klasik manipulasyon uretir. GPU gerektirmez.

    splits: sadece bu split'lerdeki kaynaklardan uret (orn. ["test","val"]).
            None ise tum havuz kullanilir.
    """
    from src.data.generators.inpaint_add import _damage_mask_path_from_manifest
    from src.data.manifest import load_manifest

    weights = weights or {"copy_move": 0.4, "splice": 0.4, "bg_replace": 0.2}
    p = np.array([weights[s] for s in SUBTYPES], dtype=float)
    p /= p.sum()

    out_dir = Path(out_root)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "gen_log.jsonl"

    df = load_manifest(manifest_path)
    pool = df[(df["label"] == "real") & (df["launder_profile"] == "clean")].reset_index(drop=True)

    # SPLIT FILTRESI -- test/val setini "besleme" icin
    # -------------------------------------------------
    # Uretim, kaynak havuzuyla ORANTILI dagilir. CarDD'nin kendi test
    # bolumu kucuk oldugu icin (%9.4) test setine cok az manipulasyon
    # duser. Olculdu: 400 uretimden test'e 38 tanesi geldi, bunun da
    # sadece 5'i bg_replace idi.
    #
    # Plan Hafta 2'nin ana ciktisi "senaryo x laundering profili" matrisi.
    # 5 ornekle bir satir doldurmak, gurultuyu sonuc diye raporlamaktir.
    #
    # Bu filtre ile SADECE test/val'e ek uretim yapilabilir; mevcut
    # dosyalar resume sayesinde atlanir, bastan uretim gerekmez:
    #     --splits test val --n 120
    if splits:
        before = len(pool)
        pool = pool[pool["split"].isin(splits)].reset_index(drop=True)
        print(f"  Split filtresi {splits}: {before} -> {len(pool)} kaynak")

    if len(pool) < 2:
        raise RuntimeError("En az 2 gercek goruntu gerekli (splice icin donor lazim).")

    # DONOR AYNI SPLIT'TEN SECILIR -- OLCULMUS BIR HATANIN DUZELTMESI
    # -----------------------------------------------------------------
    # splice ve bg_replace iki kaynak goruntu kullanir: hedef + donor.
    # Donor rastgele secilirse, train'deki bir fotograf test'teki bir
    # baskasiyla eslesebilir. build_manifest_v2 bunu sizinti sayip
    # (hakli olarak) grubu TEST'e alir. Ilk uretimde olculen sonuc:
    #
    #     116 grup catisti
    #     206 gercek goruntu train/val'den test'e suruklendi
    #     test:  splice 91 | bg_replace 41 | copy_move 19
    #     val :  splice  6 | bg_replace  2 | copy_move 30
    #
    # Yani donor kullanan senaryolar test'te yigildi, val'de yok oldu.
    # Test seti artik planlanan senaryo karisimini TEMSIL ETMIYOR ve val
    # ile bu senaryolar icin hicbir sey ayarlanamaz.
    #
    # Kok neden uretimde: sizinti korumasini manifest'e birakmak yerine
    # donoru bastan ayni split'ten secmek gerekiyordu. Boylece catisma
    # HIC olusmaz, CarDD'nin kuratorlu bolumu bozulmaz.
    rng = np.random.default_rng(seed)
    split_pools: dict[str, np.ndarray] = {
        s: np.flatnonzero((pool["split"] == s).to_numpy())
        for s in pool["split"].unique()
    }
    for s, ids in split_pools.items():
        if len(ids) < 2:
            print(f"  UYARI: '{s}' split'inde {len(ids)} goruntu var; "
                  f"donor gerektiren senaryolar bu split'te uretilemeyecek.")

    def pick_donor(target_idx: int, target_split: str) -> int | None:
        """Hedefle AYNI split'ten, hedeften farkli bir donor secer."""
        cands = split_pools.get(target_split)
        if cands is None or len(cands) < 2:
            return None
        for _ in range(10):
            j = int(cands[rng.integers(0, len(cands))])
            if j != target_idx:
                return j
        return None

    order = rng.permutation(len(pool))
    results: list[GenResult] = []
    skipped = 0
    rejected = 0
    t0 = time.time()

    for idx in order:
        if len(results) >= n:
            break
        row = pool.iloc[int(idx)]
        src_path = Path(row["path"])
        if not src_path.exists():
            skipped += 1
            continue
        img = imread(src_path)
        if img is None:
            skipped += 1
            continue

        dmg_path = _damage_mask_path_from_manifest(row["gen_params"])
        H, W = img.shape[:2]
        dmg = load_mask(dmg_path, size=(W, H)) if dmg_path and Path(dmg_path).exists() else None

        subtype = SUBTYPES[int(rng.choice(len(SUBTYPES), p=p))]
        sid = str(row["source_image_id"])
        out_path = out_dir / f"{sid}_{subtype}.png"
        mask_path = out_dir / "masks" / f"{sid}_{subtype}.png"
        if resume and out_path.exists() and mask_path.exists():
            continue

        donor_sid = ""
        if subtype == "copy_move":
            res = copy_move(img, dmg, rng)
        elif subtype == "splice":
            # DONOR SECIMI KRITIK: donor da bir kaynak goruntudur.
            # Ayni split'te kalmasi icin donor'un id'si gen_params'a yazilir
            # ve build_manifest_v2 donor'u da grup anahtarina dahil eder.
            j = pick_donor(int(idx), str(row["split"]))
            if j is None:
                skipped += 1
                continue
            drow = pool.iloc[j]
            donor = imread(drow["path"])
            if donor is None:
                skipped += 1
                continue
            dpath = _damage_mask_path_from_manifest(drow["gen_params"])
            dmask = load_mask(dpath, size=(donor.shape[1], donor.shape[0])) \
                if dpath and Path(dpath).exists() else None
            donor_sid = str(drow["source_image_id"])
            res = splice(img, donor, dmask, rng, dst_damage_mask=dmg)
        else:
            j = pick_donor(int(idx), str(row["split"]))
            if j is None:
                skipped += 1
                continue
            bg = imread(pool.iloc[j]["path"])
            if bg is None:
                skipped += 1
                continue
            donor_sid = str(pool.iloc[j]["source_image_id"])
            res = bg_replace(img, bg, rng, damage_mask=dmg)

        if res is None:
            skipped += 1
            continue
        out_img, out_mask = res

        changed = changed_fraction_in_mask(img, out_img, out_mask)
        if changed < MIN_CHANGED_IN_MASK:
            # Manipulasyon gorunmez kaldi (ayni renkli bolge, notr blend...).
            # Diske YAZMA -- bkz. MIN_CHANGED_IN_MASK aciklamasi.
            rejected += 1
            continue

        # IKINCI/UCUNCU KABUL KAPISI (W3 duzeltmesi) -- renk uyumu
        # -----------------------------------------------------------
        # copy_move/splice icin _paste_region artik color_transfer
        # uyguluyor, ama seamlessClone/color_transfer nadir girdilerde
        # (cok kucuk maske, dusuk kontrast) yine de basarisiz kalabilir.
        # diag_splice.py ile olculen dE metrigini burada da uygulayip
        # alakasiz renkli sonuclari diske yazmadan eliyoruz.
        #
        # bg_replace ayri ele alinir: MUTLAK dE degil, dogal taban
        # DEGERINE GORE ARTIS olculur (bkz. BG_MAX_COLOR_ARTIS aciklamasi)
        # -- cunku bu senaryoda arac/arka plan sinirinda dogal fotograflarda
        # bile hatiri sayilir bir renk farki OLMASI beklenir.
        if subtype in ("copy_move", "splice"):
            de = color_consistency_de(out_img, out_mask)
            if de is not None and de > MAX_COLOR_DE:
                rejected += 1
                continue
        elif subtype == "bg_replace":
            de_manip = color_consistency_de(out_img, out_mask)
            de_kaynak = color_consistency_de(img, out_mask)
            if (
                de_manip is not None
                and de_kaynak is not None
                and (de_manip - de_kaynak) > BG_MAX_COLOR_ARTIS
            ):
                rejected += 1
                continue

        leak = leak_fraction_outside_mask(img, out_img, out_mask)

        imwrite(out_path, out_img)
        save_mask(out_mask, mask_path)

        record = {
            "path": str(out_path).replace("\\", "/"),
            "mask_path": str(mask_path).replace("\\", "/"),
            "source_path": str(src_path).replace("\\", "/"),
            "manip_type": subtype,
            "donor_source_image_id": donor_sid,
            "method": "opencv",
            "seed": seed,
            "mask_area_frac": round(mask_area_frac(out_mask), 5),
            "changed_frac_in_mask": round(changed, 4),
            "leak_frac_outside_mask": round(leak, 5),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        results.append(
            GenResult(
                path=record["path"],
                label="manipulated",
                source_image_id=sid,
                manip_type=subtype,
                generator="opencv",
                mask_path=record["mask_path"],
                width=W,
                height=H,
                gen_params=record,
            )
        )

        if len(results) % 50 == 0:
            print(f"  {len(results)}/{n}  ({(time.time()-t0)/len(results):.2f} sn/goruntu)")

    counts = {s: sum(r.manip_type == s for r in results) for s in SUBTYPES}
    print(
        f"[classic] {len(results)} uretildi | {skipped} atlandi | "
        f"{rejected} kalite kapisinda reddedildi | {counts}"
    )
    if rejected > len(results):
        print(
            "  UYARI: Reddedilen ornek sayisi uretilenden fazla. Kaynak "
            "goruntulerin dokusu zayif olabilir (duz renkli bolgeler)."
        )
    return results


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="M3: klasik manipulasyon (GPU'suz)")
    ap.add_argument("--manifest", default="data/processed/manifest_v1.parquet")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", nargs="*", default=None,
                    choices=["train", "val", "test"],
                    help="Sadece bu split'lerden uret (test/val besleme icin)")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    generate(a.manifest, a.n, out_root=a.out, seed=a.seed,
             splits=a.splits, resume=not a.no_resume)


if __name__ == "__main__":
    _cli()
