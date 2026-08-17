"""
Hafta 2 manifest olusturucu -- tum katmanlari tek manifest'te birlestirir.

GIRDILER
--------
    data/raw/cardd/CarDD_COCO/**            R  gercek (kendi split'i var)
    data/raw/own_photos/anon/*.jpg          R  kendi telefon fotograflarin
    data/raw/synthetic/{model}/*.png        S  tam sentetik
    data/raw/manipulated/inpaint_add/       M1 hasar ekleme/buyutme
    data/raw/manipulated/inpaint_remove/    M2 hasar silme
    data/raw/manipulated/classic/           M3 copy-move / splice / bg_replace

CIKTI
-----
    data/processed/manifest_v2.parquet      tum katmanlar, clean profil
    data/processed/split_groups.json        hangi kaynak hangi gruba dustu

SPLIT MANTIGI -- PROJENIN EN KRITIK 40 SATIRI (plan 4.5)
---------------------------------------------------------
Tuzak 1'e karsi: bir CarDD fotografi ve ondan turetilmis TUM manipulasyonlar
ayni split'te olmali. Bunu "source_image_id'ye gore grupla" ile cozmek
YETMEZ, cunku splice iki kaynak goruntu kullanir:

    cardd_A (hedef) + cardd_B (donor)  ->  bir manipule goruntu

Bu goruntu source_image_id=cardd_A tasir. Eger cardd_B test'te, cardd_A
train'de ise, modelin train'de gordugu piksellerin bir kismi test setinden
gelmis olur. Sessiz bir sizinti.

COZUM -- IKI KADEMELI:

1. KAYNAK SPLIT'LERI DEGISMEZ. CarDD icin klasor yapisindan
   (train2017/val2017/test2017) gelir; digerleri icin id hash'inden.
   Hicbir kosulda oynamaz. Bkz. canonical_split().

2. Turetilmis bir goruntunun hedefi ile donoru FARKLI split'lerdeyse,
   o TURETILMIS GORUNTU manifest'e alinmaz. Kaynaklar yerinde kalir.

Neden kaynagi degil turetilmisi eliyoruz: kaynak fotograflar veri setinin
omurgasi ve yeniden uretilemez; turetilmis goruntuler ise bir komutla
yeniden uretilir. Ilk surumde tersi yapiliyordu (kaynaklari test'e tasi)
ve olculen bedel agirdi -- 116 catisma, 206 gercek goruntu yer degistirdi,
val setinde 2 tane bg_replace kaldi.

Dahasi, kaynaklari tasimak KENDINI BESLEYEN bir dongu yaratiyordu:
uretici bozuk manifest'i okuyup "test" sanilan bir train goruntusunden
donor seciyor, yeniden kurulumda catisma tekrar doguyordu.

Tuzak 2/3'e karsi: bu script hicbir goruntuyu yeniden boyutlandirmaz ama
cozunurluk ve format dagilimini SPLIT x LABEL bazinda RAPORLAR. W1 Bulgu
3'un (AUC 0.364) takibi buradan yapilir; ayrica scripts/run_e1_shortcut.py
bunu deneysel olarak test eder.

E6'ya karsi: flux_schnell (test-only uretici) train/val'e sizarsa script
HATA VERIR ve manifest yazilmaz.

Calistirma:
    python scripts/build_manifest_v2.py
    python scripts/build_manifest_v2.py --freeze-test   # test manifestini dondur
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generators.pipelines import TEST_ONLY_GENERATORS
from src.data.manifest import (
    check_generator_disjoint,
    check_split_leakage,
    check_unique_image_id,
    make_row,
    rows_to_manifest,
    save_manifest,
    summarize,
)

CARDD_COCO = Path("data/raw/cardd/CarDD_COCO")
COCO_SPLIT_MAP = {"train2017": "train", "val2017": "val", "test2017": "test"}
SOD_MASK_DIRS = {
    "train": Path("data/raw/cardd/CarDD_SOD/CarDD-TR/CarDD-TR-Mask"),
    "val": Path("data/raw/cardd/CarDD_SOD/CarDD-VAL/CarDD-VAL-Mask"),
    "test": Path("data/raw/cardd/CarDD_SOD/CarDD-TE/CarDD-TE-Mask"),
}
OWN_PHOTOS = Path("data/raw/own_photos/anon")
SYNTH_ROOT = Path("data/raw/synthetic")
MANIP_DIRS = {
    "inpaint_add": Path("data/raw/manipulated/inpaint_add"),
    "inpaint_remove": Path("data/raw/manipulated/inpaint_remove"),
    "classic": Path("data/raw/manipulated/classic"),
}

OUTPUT = Path("data/processed/manifest_v2.parquet")
GROUPS_OUT = Path("data/processed/split_groups.json")
FROZEN_TEST = Path("data/processed/test_manifest_frozen.parquet")

TRAIN_RATIO, VAL_RATIO = 0.70, 0.15
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Union-Find (sizinti korumasi)
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def hash_split(key: str) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    r = (h % 10000) / 10000.0
    if r < TRAIN_RATIO:
        return "train"
    if r < TRAIN_RATIO + VAL_RATIO:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Toplama
# ---------------------------------------------------------------------------


def size_of(p: Path) -> tuple[int, int] | None:
    try:
        with Image.open(p) as im:
            return im.size
    except Exception:
        return None


def collect_cardd() -> list[dict]:
    items = []
    if not CARDD_COCO.exists():
        print("  UYARI: CarDD bulunamadi, atlaniyor.")
        return items
    for folder, split in COCO_SPLIT_MAP.items():
        d = CARDD_COCO / folder
        if not d.exists():
            continue
        for p in sorted(q for q in d.rglob("*") if q.suffix.lower() in IMAGE_EXTS):
            s = size_of(p)
            if s is None:
                continue
            mdir = SOD_MASK_DIRS[split]
            dm = mdir / f"{p.stem}.png"
            items.append({
                "source_image_id": f"cardd_{p.stem}",
                "path": str(p).replace("\\", "/"),
                "label": "real",
                "manip_type": "none",
                "generator": "none",
                "mask_path": "",
                "width": s[0], "height": s[1],
                "fixed_split": split,  # CarDD'nin kendi bolumu
                "gen_params": {"damage_mask_path": str(dm).replace("\\", "/")}
                if dm.exists() else {},
            })
    return items


def collect_own() -> list[dict]:
    items = []
    if not OWN_PHOTOS.exists():
        return items
    exif_dir = OWN_PHOTOS.parent / "exif"
    for p in sorted(q for q in OWN_PHOTOS.iterdir() if q.suffix.lower() in IMAGE_EXTS):
        s = size_of(p)
        if s is None:
            continue
        meta = {}
        ej = exif_dir / f"{p.stem}.json"
        if ej.exists():
            try:
                j = json.loads(ej.read_text(encoding="utf-8"))
                meta = {
                    "damage_label": j.get("damage_label"),
                    "exif_quality": j.get("exif_quality"),
                    "camera": f"{j.get('exif',{}).get('Make','')} "
                              f"{j.get('exif',{}).get('Model','')}".strip(),
                    "original_path": str(OWN_PHOTOS.parent / "original").replace("\\", "/"),
                }
            except json.JSONDecodeError:
                pass
        items.append({
            "source_image_id": p.stem, "path": str(p).replace("\\", "/"),
            "label": "real", "manip_type": "none", "generator": "none",
            "mask_path": "", "width": s[0], "height": s[1],
            "fixed_split": None, "gen_params": meta,
        })
    return items


def collect_synthetic() -> list[dict]:
    items = []
    if not SYNTH_ROOT.exists():
        return items
    for mdir in sorted(d for d in SYNTH_ROOT.iterdir() if d.is_dir()):
        model = mdir.name
        logs = {}
        lp = mdir / "gen_log.jsonl"
        if lp.exists():
            for line in lp.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    logs[Path(r["path"]).name] = r
                except (json.JSONDecodeError, KeyError):
                    continue
        for p in sorted(q for q in mdir.iterdir() if q.suffix.lower() in IMAGE_EXTS):
            s = size_of(p)
            if s is None:
                continue
            items.append({
                "source_image_id": f"synth_{model}_{p.stem}",
                "path": str(p).replace("\\", "/"),
                "label": "fully_synthetic", "manip_type": "none",
                "generator": model, "mask_path": "",
                "width": s[0], "height": s[1],
                # Test-only ureticiler DOGRUDAN test'e sabitlenir.
                "fixed_split": "test" if model in TEST_ONLY_GENERATORS else None,
                "gen_params": logs.get(p.name, {"model": model}),
            })
    return items


def collect_manipulated() -> list[dict]:
    items = []
    for key, d in MANIP_DIRS.items():
        lp = d / "gen_log.jsonl"
        if not lp.exists():
            continue
        for line in lp.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = Path(r["path"])
            if not p.exists():
                continue
            s = size_of(p)
            if s is None:
                continue
            sid = Path(r.get("source_path", "")).stem
            sid = f"cardd_{sid}" if sid and not sid.startswith(("own_", "cardd_")) else sid
            items.append({
                "source_image_id": sid or p.stem,
                # variant_id: TURETILMIS goruntunun kendi birincil anahtari.
                # source_image_id kaynakla ORTAKTIR (split grubu icin), bu
                # yuzden image_id'yi ondan uretmek cakisma yaratir --
                # bkz. src/data/manifest.py modul basligi.
                "variant_id": f"{key}_{p.stem}",
                "path": str(p).replace("\\", "/"),
                "label": "manipulated",
                "manip_type": r.get("manip_type", key),
                "generator": r.get("model", "unknown"),
                "mask_path": r.get("mask_path", ""),
                "width": s[0], "height": s[1],
                "fixed_split": None,
                "donor": r.get("donor_source_image_id", ""),
                "gen_params": r,
            })
    return items


# ---------------------------------------------------------------------------
# Split atamasi
# ---------------------------------------------------------------------------


def canonical_split(items: list[dict]) -> dict[str, str]:
    """Her KAYNAK goruntunun degismez split'i.

    CarDD icin klasor yapisindan (train2017/val2017/test2017) gelir --
    kuratorlu ve sabittir. Diger kaynaklar (kendi fotograflarin, sentetik)
    icin id'nin hash'inden deterministik olarak turetilir.

    Bu deger HICBIR KOSULDA degismez. Bir onceki surumde donor catismasi
    olunca gruplar test'e tasiniyordu; bu, CarDD'nin kuratorlu bolumunu
    bozuyordu ve daha kotusu KENDINI BESLIYORDU:

        classic_manip bozuk manifest_v2'yi okur
        -> "test" sanip test'ten donor secer
        -> ama o goruntu aslinda train2017'de
        -> yeniden kurulumda yine catisma cikar

    Artik kaynak split'leri sabit; catismayi TURETILMIS goruntuyu eleyerek
    cozuyoruz (bkz. assign_splits).
    """
    out: dict[str, str] = {}
    for it in items:
        sid = it["source_image_id"]
        if sid in out:
            continue
        out[sid] = it["fixed_split"] or hash_split(sid)
    return out


def assign_splits(
    items: list[dict],
) -> tuple[dict[str, str], dict[str, list[str]], list[int]]:
    """Split atar ve split-asiri donor kullanan ogeleri ELER.

    Doner: (sid -> split), (grup -> uyeler), (elenecek oge indeksleri)

    TASARIM KARARI -- catismayi kim oder?
    -------------------------------------
    Bir splice iki kaynak kullanir. Kaynaklar farkli split'lerdeyse bir
    sorun var demektir. Iki secenek:

      (a) Kaynaklari ayni split'e tasi   <- ESKI DAVRANIS
          Bedeli: CarDD'nin kuratorlu bolumu bozulur, yuzlerce gercek
          goruntu yer degistirir, senaryo dagilimi carpilir. Olculdu:
          116 catisma -> 206 gercek goruntu test'e suruklendi, val'de
          sadece 2 bg_replace kaldi.

      (b) TURETILMIS goruntuyu ele      <- YENI DAVRANIS
          Bedeli: birkac uretilmis ornek kaybedilir (yenisi uretilebilir).
          Kazanci: kaynak split'leri HIC oynamaz.

    (b) dogru olan: kaynak fotograflar veri setinin omurgasi, turetilmis
    goruntuler ise yeniden uretilebilir. Pahali olani korumak gerekir.
    """
    canon = canonical_split(items)

    uf = UnionFind()
    drop: list[int] = []
    cross = 0

    for i, it in enumerate(items):
        sid = it["source_image_id"]
        uf.find(sid)
        donor = it.get("donor", "")
        if not donor:
            continue
        if donor not in canon:
            continue  # donor manifest'te yok; grup baglantisi kurulamaz
        if canon[donor] != canon[sid]:
            # Split-asiri donor: bu TURETILMIS goruntu manifest'e ALINMAZ.
            drop.append(i)
            cross += 1
            continue
        uf.union(sid, donor)

    dropped = set(drop)
    groups: dict[str, list[str]] = defaultdict(list)
    for i, it in enumerate(items):
        if i in dropped:
            continue
        groups[uf.find(it["source_image_id"])].append(it["source_image_id"])

    sid_split = dict(canon)

    if cross:
        print(f"  NOT: {cross} turetilmis goruntu split-asiri donor kullaniyordu "
              f"-> manifest'e ALINMADI (kaynak split'leri korundu).")
        print("       Bunlari geri kazanmak icin classic_manip'i TEMIZ bir "
              "manifest'le yeniden calistir.")

    # Ayni gruptaki tum uyelerin split'i ayni mi? (kendi kendini denetleme)
    for root, members in groups.items():
        splits = {sid_split[m] for m in members}
        if len(splits) > 1:
            raise RuntimeError(
                f"IC TUTARSIZLIK: grup {root} birden fazla split iceriyor: {splits}. "
                f"Bu bir kod hatasidir, veri hatasi degil."
            )

    return sid_split, {k: sorted(set(v)) for k, v in groups.items()}, drop


# ---------------------------------------------------------------------------
# Raporlama
# ---------------------------------------------------------------------------


def resolution_report(items: list[dict], sid_split: dict[str, str]) -> None:
    """Plan 4.5 Tuzak 2/3'un teshis raporu.

    Real ve fake sinifin cozunurluk/format dagilimlari birbirinden UZAKSA
    model forensic iz yerine bu kestirmeyi ogrenir. W1'de AUC 0.364
    cikmasinin hipotezi tam olarak buydu."""
    by_label: dict[str, list[int]] = defaultdict(list)
    fmt: dict[str, Counter] = defaultdict(Counter)
    for it in items:
        by_label[it["label"]].append(max(it["width"], it["height"]))
        fmt[it["label"]][Path(it["path"]).suffix.lower()] += 1

    print("\nCOZUNURLUK / FORMAT TESHISI (plan 4.5 Tuzak 2 ve 3)")
    print(f"{'label':<18}{'n':<8}{'uzun kenar medyan':<20}{'min-max':<16}format")
    print("-" * 82)
    medians = {}
    for lab, vals in sorted(by_label.items()):
        vals_sorted = sorted(vals)
        med = vals_sorted[len(vals_sorted) // 2]
        medians[lab] = med
        print(f"{lab:<18}{len(vals):<8}{med:<20}"
              f"{f'{vals_sorted[0]}-{vals_sorted[-1]}':<16}{dict(fmt[lab])}")

    if "real" in medians:
        for lab, med in medians.items():
            if lab == "real":
                continue
            ratio = med / max(1, medians["real"])
            if ratio < 0.7 or ratio > 1.4:
                print(f"\n  >>> UYARI: '{lab}' medyan cozunurlugu 'real'in "
                      f"{ratio:.2f} kati.")
                print("  >>> Bu, plan 4.5 Tuzak 2'nin ta kendisi. Laundering "
                      "katmani bunu kismen kapatir;")
                print("  >>> scripts/run_e1_shortcut.py ile OLC, tahmin etme.")

    all_fmt = set().union(*[set(c) for c in fmt.values()]) if fmt else set()
    if len(all_fmt) > 1:
        print("\n  >>> NOT: Katmanlar farkli dosya formatlarinda "
              f"({sorted(all_fmt)}).")
        print("  >>> Bu NORMAL: ham cikti PNG, gercek foto JPEG. Egitim ve")
        print("  >>> degerlendirme SADECE laundered (hepsi JPEG) kopyalar")
        print("  >>> uzerinden yapilmalidir -- scripts/apply_laundering.py.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTPUT))
    ap.add_argument("--freeze-test", action="store_true",
                    help="Test setini ayri bir dosyaya dondurup hash'le")
    a = ap.parse_args()

    print("=" * 62)
    print("MANIFEST V2 -- tum katmanlar")
    print("=" * 62)

    items: list[dict] = []
    for name, fn in [
        ("CarDD (R)", collect_cardd),
        ("Kendi fotograflarin (R)", collect_own),
        ("Tam sentetik (S)", collect_synthetic),
        ("Manipulasyon (M)", collect_manipulated),
    ]:
        got = fn()
        print(f"  {name:<26} {len(got)}")
        items.extend(got)

    if not items:
        print("\nHATA: hicbir goruntu bulunamadi.")
        sys.exit(1)

    sid_split, groups, drop = assign_splits(items)
    if drop:
        dropped = {items[i]["manip_type"] for i in drop}
        print(f"  Elenen oge: {len(drop)} ({sorted(dropped)})")
        keep = set(range(len(items))) - set(drop)
        items = [items[i] for i in sorted(keep)]

    df = rows_to_manifest([
        make_row(
            source_image_id=it["source_image_id"],
            variant_id=it.get("variant_id"),
            path=it["path"],
            label=it["label"],
            manip_type=it["manip_type"],
            generator=it["generator"],
            mask_path=it["mask_path"],
            width=it["width"], height=it["height"],
            split=sid_split[it["source_image_id"]],
            launder_profile="clean",
            gen_params=it["gen_params"],
        )
        for it in items
    ])

    print("\n" + "=" * 62)
    print("DOGRULAMA")
    print("=" * 62)

    problems = check_split_leakage(df)
    problems += check_generator_disjoint(df, TEST_ONLY_GENERATORS)
    problems += check_unique_image_id(df)
    if problems:
        print("DOGRULAMA BASARISIZ -- manifest KAYDEDILMEDI:")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    print("Split sizintisi        : TEMIZ")
    print(f"Generator-disjoint     : TEMIZ (test-only: {TEST_ONLY_GENERATORS})")
    print(f"image_id benzersiz     : TEMIZ ({df['image_id'].nunique()} kimlik)")

    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"Birlestirilmis grup    : {len(multi)} (splice/bg_replace donor baglari)")

    for split in ("train", "val", "test"):
        labs = set(df[df["split"] == split]["label"])
        print(f"  {split:<6} n={len(df[df['split']==split]):<6} etiketler={labs or '{}'}")
        if split == "test" and len(labs) < 2:
            print("    >>> UYARI: Test setinde tek sinif var, metrik hesaplanamaz.")

    resolution_report(items, sid_split)

    print("\nDAGILIM")
    print(summarize(df).to_string(index=False))

    save_manifest(df, a.out)
    GROUPS_OUT.parent.mkdir(parents=True, exist_ok=True)
    GROUPS_OUT.write_text(json.dumps(multi, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Grup haritasi kaydedildi: {GROUPS_OUT}")

    if a.freeze_test:
        test_df = df[df["split"] == "test"].reset_index(drop=True)
        save_manifest(test_df, FROZEN_TEST)
        h = hashlib.sha256(
            "".join(sorted(test_df["image_id"].astype(str))).encode()
        ).hexdigest()
        (FROZEN_TEST.with_suffix(".sha256")).write_text(h, encoding="utf-8")
        print("\n" + "=" * 62)
        print("TEST SETI DONDURULDU")
        print("=" * 62)
        print(f"  {FROZEN_TEST}  ({len(test_df)} satir)")
        print(f"  sha256: {h}")
        print("  Bu dosyayi commit'le ve Hafta 5'e kadar BIR DAHA ACMA.")
        print("  Test setine bakarak model secmek = kendini kandirmak (plan 4.5 Tuzak 5).")

    print("\nSonraki adim: python scripts/apply_laundering.py")


if __name__ == "__main__":
    main()
