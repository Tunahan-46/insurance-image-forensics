"""
Laundering uygulayici (plan 4.4).

Manifest'teki her `clean` satiri icin secilen profillerin laundered
kopyalarini uretir ve manifest'e YENI SATIRLAR ekler.

    1 kaynak goruntu  ->  5 satir (clean + 4 profil)

KURALLAR
--------
1. Train/val icin sadece TRAIN_AUGMENT_PROFILES uretilir (disk tasarrufu;
   egitim sirasinda augmentation zaten rastgele profil secebilir).
   TEST icin BES PROFILIN HEPSI uretilir -- sonuc tablosu senaryo x profil
   matrisidir ve eksik hucre birakmak raporun degerini dusurur.

2. Maskeler goruntuyle ayni geometrik donusumden gecer (src.data.launder.
   launder_mask). Maske JPEG'lenmez, PNG kalir.

3. `clean` profili de yeniden kaydedilir (q95 JPEG). Bu, plan 4.5 Tuzak 3'u
   (format kestirme yolu) kapatan adimdir: PNG sentetikler ve JPEG gercekler
   ayni encoder'dan gecer. EGITIM VE DEGERLENDIRME BU KOPYALAR UZERINDEN
   YAPILIR, ham dosyalar uzerinden DEGIL.

   TEK ISTISNA: L1 metadata dedektoru. O, orijinal yuklenen dosyayi okur
   (plan 7.1). Manifest'in gen_params alaninda orijinal yol saklanir.

Cikti:
    data/processed/laundered/{profil}/{image_id}.jpg
    data/processed/laundered/{profil}/masks/{image_id}.png
    data/processed/manifest_v2_laundered.parquet

Calistirma:
    python scripts/apply_laundering.py
    python scripts/apply_laundering.py --profiles whatsapp aggressive --splits test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.launder import (
    PROFILE_NAMES,
    TRAIN_AUGMENT_PROFILES,
    launder_file,
)
from src.data.manifest import (
    add_row,
    check_split_leakage,
    load_manifest,
    new_manifest,
    save_manifest,
    summarize,
)

IN_MANIFEST = "data/processed/manifest_v2.parquet"
OUT_MANIFEST = "data/processed/manifest_v2_laundered.parquet"
OUT_ROOT = Path("data/processed/laundered")


def profiles_for_split(split: str, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return list(PROFILE_NAMES) if split == "test" else list(TRAIN_AUGMENT_PROFILES)


def main() -> None:
    ap = argparse.ArgumentParser(description="Laundering profillerini uygula")
    ap.add_argument("--manifest", default=IN_MANIFEST)
    ap.add_argument("--out-manifest", default=OUT_MANIFEST)
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--profiles", nargs="*", default=None, choices=list(PROFILE_NAMES))
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="Test amacli: ilk N kaynak")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if not Path(a.manifest).exists():
        print(f"HATA: {a.manifest} yok. Once: python scripts/build_manifest_v2.py")
        sys.exit(1)

    df = load_manifest(a.manifest)
    base = df[df["launder_profile"] == "clean"].reset_index(drop=True)
    base = base[base["split"].isin(a.splits)].reset_index(drop=True)
    if a.limit:
        base = base.head(a.limit)

    print(f"{len(base)} kaynak goruntu | split'ler: {a.splits}")
    if a.profiles:
        print(f"Profiller (elle secildi): {a.profiles}")
    else:
        print(f"Profiller: test -> {list(PROFILE_NAMES)}")
        print(f"           train/val -> {list(TRAIN_AUGMENT_PROFILES)}")

    out_root = Path(a.out_root)
    out_df = new_manifest()
    counts: Counter[str] = Counter()
    errors = 0
    t0 = time.time()

    # MASKE KAYBI TAKIBI (W3) -- geometri normalizasyonu merkezden KARE
    # kirptigi icin (bkz. src.data.launder.NORMALIZE_EDGE) kenarda kalan bir
    # manipulasyon bolgesi kirpilip gidebilir. Zemin gercegi bos kalirsa
    # o ornek "manipule" etiketli ama gosterilecek yeri olmayan bir kayda
    # doner ve piksel-F1'i sessizce bozar. Sayiyi topluyoruz ki sessiz
    # kalmasin.
    mask_lost: list[str] = []
    mask_shrunk: list[tuple[str, float]] = []

    for i, row in base.iterrows():
        src = Path(row["path"])
        if not src.exists():
            errors += 1
            continue

        mask_src = row["mask_path"] if row["mask_path"] else None
        if mask_src and not Path(mask_src).exists():
            mask_src = None

        gp = row["gen_params"]
        try:
            gp_dict = json.loads(gp) if isinstance(gp, str) else dict(gp or {})
        except (json.JSONDecodeError, TypeError):
            gp_dict = {}
        # L1 metadata dedektoru icin orijinal yolu KAYBETME (plan 7.1).
        gp_dict["original_path"] = str(src).replace("\\", "/")

        for profile in profiles_for_split(str(row["split"]), a.profiles):
            image_id = f"{row['source_image_id']}__{profile}"
            dst = out_root / profile / f"{image_id}.jpg"
            mask_dst = (out_root / profile / "masks" / f"{image_id}.png") if mask_src else None

            if dst.exists() and not a.overwrite and (mask_dst is None or mask_dst.exists()):
                info = {
                    "dst": str(dst).replace("\\", "/"),
                    "new_size": Image.open(dst).size,
                    "mask_dst": str(mask_dst).replace("\\", "/") if mask_dst else "",
                    "save_quality": None,
                }
            else:
                try:
                    info = launder_file(
                        src, dst, profile, mask_src=mask_src, mask_dst=mask_dst
                    )
                except Exception as e:
                    print(f"  [hata] {src.name} / {profile}: {e}")
                    errors += 1
                    continue

            if profile == "clean" and info["mask_dst"]:
                try:
                    import numpy as _np

                    before = _np.asarray(Image.open(mask_src).convert("L")) > 127
                    after = _np.asarray(Image.open(info["mask_dst"]).convert("L")) > 127
                    f0, f1 = before.mean(), after.mean()
                    if f1 < 1e-5:
                        mask_lost.append(str(row["source_image_id"]))
                    elif f0 > 0 and f1 / f0 < 0.5:
                        mask_shrunk.append((str(row["source_image_id"]), f1 / f0))
                except Exception:
                    pass

            out_df = add_row(
                out_df,
                source_image_id=str(row["source_image_id"]),
                path=info["dst"],
                label=str(row["label"]),
                manip_type=str(row["manip_type"]),
                generator=str(row["generator"]),
                mask_path=info["mask_dst"],
                width=int(info["new_size"][0]),
                height=int(info["new_size"][1]),
                split=str(row["split"]),
                launder_profile=profile,
                gen_params={**gp_dict, "launder_save_quality": info["save_quality"]},
            )
            counts[profile] += 1

        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(base)} kaynak  ({el:.0f} sn, {sum(counts.values())} cikti)")

    print("\n" + "=" * 62)
    print("DOGRULAMA")
    print("=" * 62)

    if len(out_df) == 0:
        print("HATA: hicbir cikti uretilmedi.")
        sys.exit(1)

    problems = check_split_leakage(out_df)
    if problems:
        print("SIZINTI -- manifest KAYDEDILMEDI:")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    print("Split sizintisi: TEMIZ")
    print(f"Hata/atlanan   : {errors}")
    print(f"Profil basina  : {dict(counts)}")

    print("\nMASKE KORUNUMU (kare kirpma sonrasi)")
    if mask_lost:
        print(f"  >>> {len(mask_lost)} maske TAMAMEN kayboldu -- bu ornekler")
        print("      'manipule' etiketli ama zemin gercegi bos. Ornekler:")
        for s in mask_lost[:10]:
            print(f"        {s}")
        print("      Cozum: bu kayitlari manifest'ten dusur veya yeniden uret.")
    else:
        print("  Tamamen kaybolan maske: 0")
    if mask_shrunk:
        print(f"  Alani yariya inen maske: {len(mask_shrunk)}")
        for s, r in sorted(mask_shrunk, key=lambda x: x[1])[:5]:
            print(f"        {s}  (kalan %{100*r:.0f})")
    else:
        print("  Alani yariya inen maske: 0")

    # Test setinde her profil x etiket hucresi dolu mu?
    test = out_df[out_df["split"] == "test"]
    if len(test) > 0:
        piv = test.groupby(["launder_profile", "label"]).size().unstack(fill_value=0)
        print("\nTEST SETI: profil x etiket matrisi")
        print(piv.to_string())
        empty = (piv == 0).sum().sum()
        if empty:
            print(f"\n  >>> UYARI: {empty} hucre bos. Sonuc tablonda delik olacak.")

    print("\nDAGILIM")
    print(summarize(out_df).to_string(index=False))

    save_manifest(out_df, a.out_manifest)
    print(f"\nToplam {len(out_df)} degerlendirme ornegi.")
    print("Sonraki adim: python scripts/run_e1_shortcut.py")


if __name__ == "__main__":
    main()
