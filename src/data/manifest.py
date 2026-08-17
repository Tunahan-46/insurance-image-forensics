"""
Veri seti manifesti.

KURAL: Klasör yapısına asla güvenme. Her görüntünün tüm bilgisi (etiket,
kaynak, generator, manipülasyon tipi, maske yolu, split, laundering profili)
bu tek parquet/csv dosyasında tutulur. Yeni bir görüntü eklemenin tek yolu
buraya bir satır eklemektir.

Şema (her satır = bir görüntü varyantı, yani orijinal + her laundering
profili ayrı bir satırdır):

    image_id        str    Benzersiz kimlik = "{variant_id}__{profil}",
                            örn. "cardd_0001__whatsapp" veya
                            "cardd_0001_copy_move__whatsapp"
    source_image_id str    Türetildiği KÖK görüntünün kimliği (split
                            sızıntısını önlemek için kritik — bkz. 4.5)
    path            str    Dosya yolu (repo köküne göre relatif)
    label           str    "real" | "fully_synthetic" | "manipulated"
    manip_type      str    "none" | "inpaint_add" | "inpaint_remove" |
                            "copy_move" | "splice" | "bg_replace" vb.
    generator       str    "none" | "sd15" | "sdxl" | "flux_schnell" | ...
    mask_path       str    Manipülasyon maskesi yolu, yoksa "" (real/fully
                            synthetic için genelde boş ya da tüm-0 maske)
    launder_profile str    "clean" | "whatsapp" | "screenshot" |
                            "double_jpeg" | "aggressive"
    split           str    "train" | "val" | "test"
    width, height   int    Piksel boyutu
    created_at      str    ISO zaman damgası (üretim izlenebilirliği için)
    gen_params      str    JSON string: prompt/seed/steps/guidance (varsa)

VARIANT_ID ve SOURCE_IMAGE_ID AYNI ŞEY DEĞİLDİR
------------------------------------------------
`source_image_id` bir GRUPLAMA anahtarıdır: cardd_0001 fotoğrafı ve ondan
türetilen her manipülasyon aynı değeri taşır, çünkü hepsi aynı split'te
kalmak zorundadır (plan 4.5, Tuzak 1).

`image_id` bir BİRİNCİL ANAHTARdır: her satır benzersiz olmalıdır.
İkisini birbirine eşitlemek şu sessiz hatayı üretiyordu:

    cardd_0001 (real)             -> image_id "cardd_0001__clean"
    cardd_0001_copy_move (manip)  -> image_id "cardd_0001__clean"   <-- AYNI

apply_laundering.py dosya adını bu kimlikten ürettiği için manipüle
görüntü, gerçek görüntünün laundered kopyasının ÜZERİNE yazıyordu:
manifest iki satır gösterir, diskte tek dosya vardır, etiketlerden biri
yalandır. Bu yüzden türetilmiş satırlar ayrıca bir `variant_id` taşır
(dosya adından gelen benzersiz kimlik) ve check_unique_image_id()
manifest diske yazılmadan önce bunu doğrular.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MANIFEST_COLUMNS = [
    "image_id",
    "source_image_id",
    "path",
    "label",
    "manip_type",
    "generator",
    "mask_path",
    "launder_profile",
    "split",
    "width",
    "height",
    "created_at",
    "gen_params",
]

VALID_LABELS = {"real", "fully_synthetic", "manipulated"}
VALID_SPLITS = {"train", "val", "test"}
VALID_LAUNDER = {"clean", "whatsapp", "screenshot", "double_jpeg", "aggressive"}


def new_manifest() -> pd.DataFrame:
    """Boş, doğru şemalı bir manifest DataFrame'i döner."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in MANIFEST_COLUMNS})


def make_image_id(variant_id: str, launder_profile: str) -> str:
    return f"{variant_id}__{launder_profile}"


def variant_id_of(image_id: str) -> str:
    """image_id'den varyant kimliğini geri çıkarır.

    Profil adlarının hiçbirinde "__" yoktur (VALID_LAUNDER), bu yüzden
    sağdan tek bölme güvenlidir."""
    return str(image_id).rsplit("__", 1)[0]


def make_row(
    *,
    source_image_id: str,
    path: str | Path,
    label: str,
    width: int,
    height: int,
    manip_type: str = "none",
    generator: str = "none",
    mask_path: str = "",
    launder_profile: str = "clean",
    split: str = "train",
    gen_params: dict | None = None,
    variant_id: str | None = None,
) -> dict:
    """Doğrulanmış tek bir manifest satırını sözlük olarak döner.

    `variant_id` verilmezse `source_image_id` kullanılır — kaynak
    görüntüler için doğru davranış budur. TÜRETİLMİŞ görüntüler
    (manipülasyon, sentetik) kendi benzersiz kimliklerini vermelidir,
    aksi halde image_id kaynakla çakışır (bkz. modül başlığı)."""
    if label not in VALID_LABELS:
        raise ValueError(f"Geçersiz label: {label}. Beklenen: {VALID_LABELS}")
    if split not in VALID_SPLITS:
        raise ValueError(f"Geçersiz split: {split}. Beklenen: {VALID_SPLITS}")
    if launder_profile not in VALID_LAUNDER:
        raise ValueError(
            f"Geçersiz launder_profile: {launder_profile}. Beklenen: {VALID_LAUNDER}"
        )

    return {
        "image_id": make_image_id(variant_id or source_image_id, launder_profile),
        "source_image_id": source_image_id,
        "path": str(path),
        "label": label,
        "manip_type": manip_type,
        "generator": generator,
        "mask_path": mask_path,
        "launder_profile": launder_profile,
        "split": split,
        "width": width,
        "height": height,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gen_params": json.dumps(gen_params or {}, ensure_ascii=False),
    }


def rows_to_manifest(rows: list[dict]) -> pd.DataFrame:
    """Satır sözlüklerinden tek seferde manifest kurar.

    NEDEN: `add_row` her çağrıda pd.concat yapar, yani N satır eklemek
    O(N^2) kopyalamadır. 14.800 satırlık laundering manifesti bu yüzden
    dakikalarca CPU yakıyordu. Toplu üretimde her zaman bu fonksiyon
    kullanılmalı; `add_row` yalnızca birkaç satırlık işler içindir."""
    if not rows:
        return new_manifest()
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def add_row(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Manifeste doğrulanmış tek bir satır ekler ve yeni DataFrame'i döner.

    Döngü içinde KULLANMA -- bkz. rows_to_manifest."""
    return pd.concat(
        [df, pd.DataFrame([make_row(**kwargs)], columns=MANIFEST_COLUMNS)],
        ignore_index=True,
    )


def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"Manifest kaydedildi: {path} ({len(df)} satır)")


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def check_split_leakage(df: pd.DataFrame) -> list[str]:
    """KRİTİK KONTROL (bkz. plan 4.5, Tuzak 1): aynı source_image_id
    birden fazla split'te görünüyor mu? Görünüyorsa split sızıntısı var
    demektir — proje bu kontrolü geçmeden hiçbir sonuç raporlanmamalı."""
    problems = []
    grouped = df.groupby("source_image_id")["split"].nunique()
    leaked = grouped[grouped > 1]
    if len(leaked) > 0:
        problems.append(
            f"SIZINTI: {len(leaked)} source_image_id birden fazla split'te "
            f"bulunuyor. Örnekler: {list(leaked.index[:5])}"
        )
    return problems


def check_unique_image_id(df: pd.DataFrame) -> list[str]:
    """KRİTİK KONTROL: image_id manifest'in birincil anahtarıdır.

    Çakışma sessiz veri bozulmasıdır: apply_laundering.py dosya adını
    image_id'den ürettiği için iki satır aynı diskteki dosyayı gösterir
    ve etiketlerden biri yalan olur. Ayrıca --freeze-test hash'i
    image_id'lerden hesaplandığı için dondurulmuş test seti de
    doğrulanamaz hale gelir."""
    problems = []
    dup = df["image_id"].value_counts()
    dup = dup[dup > 1]
    if len(dup) > 0:
        ex = list(dup.index[:5])
        problems.append(
            f"CAKISMA: {len(dup)} image_id birden fazla satirda. "
            f"Ornekler: {ex}. Turetilmis satirlar variant_id vermelidir."
        )
    return problems


def check_generator_disjoint(df: pd.DataFrame, test_only_generators: set[str]) -> list[str]:
    """KRİTİK KONTROL (bkz. plan E6): test setindeki generator'lar train/val'de
    hiç görünmemeli. Bu, 'görülmemiş generator'a genelleme' deneyinin
    geçerliliğini garanti eder."""
    problems = []
    trainval_gens = set(df[df["split"].isin(["train", "val"])]["generator"].unique())
    overlap = trainval_gens & test_only_generators
    if overlap:
        problems.append(
            f"SIZINTI: test-only generator(lar) train/val'de de görünüyor: {overlap}"
        )
    return problems


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Hızlı bir 'veri setinde ne var' özeti — her hafta Cuma raporunda kullan."""
    return (
        df.groupby(["split", "label", "manip_type", "launder_profile"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "label", "manip_type"])
    )


def file_hash(path: str | Path) -> str:
    """Dosya içeriğinin sha256'sı — dedup ve reproducibility için."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    # Sanity check: manifest oluştur, birkaç satır ekle, sızıntı kontrolü yap.
    df = new_manifest()
    df = add_row(
        df,
        source_image_id="cardd_0001",
        path="data/raw/cardd/0001.jpg",
        label="real",
        width=1024,
        height=768,
        split="train",
        launder_profile="clean",
    )
    df = add_row(
        df,
        source_image_id="cardd_0001",
        path="data/processed/cardd_0001_whatsapp.jpg",
        label="real",
        width=1024,
        height=768,
        split="train",
        launder_profile="whatsapp",
    )
    # Kasıtlı sızıntı örneği: aynı source_image_id farklı split'te
    df_bad = add_row(
        df,
        source_image_id="cardd_0001",
        path="data/processed/cardd_0001_test.jpg",
        label="real",
        width=1024,
        height=768,
        split="test",
        launder_profile="clean",
    )

    # Türetilmiş görüntü: AYNI source_image_id (split grubu korunur) ama
    # KENDİ variant_id'si (image_id çakışmaz).
    df = add_row(
        df,
        source_image_id="cardd_0001",
        variant_id="cardd_0001_copy_move",
        path="data/raw/manipulated/classic/cardd_0001_copy_move.png",
        label="manipulated",
        manip_type="copy_move",
        width=1024,
        height=768,
        split="train",
        launder_profile="clean",
    )

    print(summarize(df))
    print("\n--- Sızıntı kontrolü (temiz df) ---")
    print(check_split_leakage(df) or "Sızıntı yok, temiz.")
    print("\n--- Sızıntı kontrolü (kasıtlı bozuk df) ---")
    print(check_split_leakage(df_bad))
    print("\n--- image_id benzersizliği ---")
    print(check_unique_image_id(df) or "Çakışma yok, temiz.")
    assert not check_unique_image_id(df), "variant_id çakışmayı önlemeliydi"

    # Toplu kurulum tek satırlık kurulumla aynı sonucu vermeli.
    bulk = rows_to_manifest([
        make_row(source_image_id="x", path="a.jpg", label="real", width=1, height=1),
        make_row(source_image_id="x", variant_id="x_splice", path="b.jpg",
                 label="manipulated", manip_type="splice", width=1, height=1),
    ])
    assert list(bulk.columns) == MANIFEST_COLUMNS, "toplu kurulum şemayı bozdu"
    assert not check_unique_image_id(bulk)

    save_manifest(df, "data/processed/manifest_sanity_check.parquet")
    loaded = load_manifest("data/processed/manifest_sanity_check.parquet")
    assert len(loaded) == len(df), "Kaydet/yükle round-trip başarısız!"
    print("\nmanifest.py sanity check OK")
