"""CLIP ViT-L/14 embedding cikarma ve onbellekleme (plan Hafta 3, E3).

NEDEN AYRI BIR MODUL VE NEDEN ONBELLEK
--------------------------------------
Plan E3 recetesi: "CLIP ViT-L/14 dondur -> tum goruntuler icin 768-d embedding
cikar, .npy cache'le (BIR KEZ) -> LogisticRegression grid search -> val'de
threshold sec -> Platt ile kalibre et."

"Bir kez" vurgusu onemli: 17.176 laundered goruntu icin embedding cikarmak
T4'te ~20-30 dakika, CPU'da saatler surer. Ama grid search, threshold secimi,
kalibrasyon ve E5/E6 varyantlari ayni embedding'ler uzerinde DAKIKALAR icinde
donuyor. Backbone dondurulmus oldugu icin embedding'ler degismez -- bir kez
cikar, defalarca kullan.

SHARD'LI YAZIM
--------------
Cikti tek bir dev .npy degil, N'lik parcalar halinde yazilir. Sebep tecrubi:
W2'de Kaggle oturumu tarayici cokmesi yuzunden sifirlandi ve /kaggle/working
tamamen silindi. Shard'li yazimda oturum koparsa yalnizca son parca kaybolur;
yeniden calistirinca kalinan yerden devam eder (`--resume`, varsayilan acik).

HIZALAMA SOZLESMESI
-------------------
Her shard yaninda bir `.ids.json` yazilir. Embedding satiri i, o dosyadaki
i'inci image_id'ye aittir. `load_embeddings()` bu id listesini manifest ile
yeniden hizalar -- satir sirasina ASLA guvenilmez, cunku manifest yeniden
uretildiginde sira degisebilir.

ONISLEME
--------
CLIP'in kendi onislemesi kullanilir (224x224, CLIP normalizasyonu). Kendi
resize'imizi yazmiyoruz: Ojha vd. paradigmasinin tum ustunlugu ozellik
uzayinin DOKUNULMAMIS olmasindan geliyor; farkli bir onisleme, egitim
dagilimindan kaymak demek.

NOT (plan 4.5 Tuzak 2): CLIP zaten her seyi 224x224'e indirir, yani
cozunurluk kestirme yolu embedding seviyesinde KISMEN kapanir. "Kismen",
cunku 512px bir goruntuyu 224'e indirmek ile 4032px bir goruntuyu 224'e
indirmek ayni doku istatistigini vermez (yeniden ornekleme artefakti farkli).
E1_shortcut'in px8/px32 problari bu farki zaten olcuyor.

Calistirma:
    python -m src.features.clip_embed --manifest data/processed/manifest_v2_laundered.parquet
    python -m src.features.clip_embed --split test --batch 64 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "openai/clip-vit-large-patch14"
DEFAULT_CACHE = Path("data/processed/clip_cache")
SHARD = 2000  # goruntu/parca -- ~6 MB float32, oturum koparsa kabul edilebilir kayip
EMBED_DIM = 768  # ViT-L/14 projeksiyon boyutu


# ---------------------------------------------------------------------------
# Model yukleme
# ---------------------------------------------------------------------------

def load_clip(model_name: str = DEFAULT_MODEL, device: str = "cuda"):
    """CLIP goruntu kodlayicisini DONDURULMUS olarak yukler.

    torch/transformers import'lari fonksiyon icinde: bu modul GPU'suz bir
    makinede sadece `load_embeddings()` icin import edilebilsin diye
    (projedeki genel kural, bkz. src/data/generators/pipelines.py)."""
    import torch
    from transformers import CLIPImageProcessor, CLIPModel

    if device == "cuda" and not torch.cuda.is_available():
        print("UYARI: cuda istendi ama bulunamadi -> cpu'ya dusuluyor.")
        device = "cpu"

    model = CLIPModel.from_pretrained(model_name)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)  # DONDURULMUS -- E3'un tum onermesi bu

    proc = CLIPImageProcessor.from_pretrained(model_name)
    return model, proc, device


# ---------------------------------------------------------------------------
# Cikarma
# ---------------------------------------------------------------------------

def _image_embeds(model, inputs):
    """CLIP goruntu embedding'ini SURUM BAGIMSIZ sekilde alir.

    NEDEN BU SARMALAYICI VAR
    ------------------------
    `CLIPModel.get_image_features()` uzun sure duz bir tensor donduruyordu.
    Yeni transformers surumlerinde ayni cagri, tensor yerine bir cikti
    NESNESI (BaseModelOutputWithPooling) donduruyor ve kod

        AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'detach'

    ile patliyor. Kaggle notebook'u `pip install transformers` ile HER ZAMAN
    en son surumu kurdugu icin bu, ortamdan ortama degisen ve sessizce geri
    gelebilecek bir kirilma noktasi. Surumu sabitlemek yerine ciktinin
    kendisine bakiyoruz -- boylece hem eski hem yeni surumde calisir.

    Uc olasi donus bicimi ele aliniyor:
      1. Duz tensor                 -> dogrudan kullan (eski davranis)
      2. .image_embeds tasiyan nesne -> projeksiyon zaten uygulanmis
      3. .pooler_output tasiyan nesne -> projeksiyon UYGULANMAMIS; ViT-L/14'un
         1024-d havuz ciktisini visual_projection ile 768'e indirmek BIZE
         dusuyor. Bu adim atlanirsa sessizce yanlis boyutlu (ve yanlis
         uzaydaki) vektorler yazilirdi -- asagidaki boyut kontrolu bunu
         imkansiz kilar.
    """
    import torch

    out = model.get_image_features(**inputs)

    if torch.is_tensor(out):
        feats = out
    elif getattr(out, "image_embeds", None) is not None:
        feats = out.image_embeds
    elif getattr(out, "pooler_output", None) is not None:
        feats = model.visual_projection(out.pooler_output)
    else:
        raise TypeError(
            f"CLIP ciktisi taninmadi: {type(out)}. "
            "transformers surumu bir kez daha degismis olabilir."
        )

    # SESSIZ HATA KAPISI: yanlis boyutlu embedding, E3'te anlamsiz ama
    # 'calisir gorunen' sonuclar uretirdi. Burada durmasi tercih edilir.
    if feats.shape[-1] != EMBED_DIM:
        raise ValueError(
            f"Beklenen embedding boyutu {EMBED_DIM}, gelen {feats.shape[-1]}. "
            "Yanlis model ya da eksik projeksiyon."
        )
    return feats


def _read_rgb(path: str | Path):
    """Unicode-guvenli okuma + RGB'ye cevirme.

    cv2.imread DOGRUDAN cagrilmaz: proje yolu 'Masaustu' iceriyor ve OpenCV
    Windows'ta ASCII disi yollari okuyamiyor (bkz. src/data/imageio.py)."""
    import cv2

    from src.data.imageio import imread

    bgr = imread(path)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def embed_paths(
    paths: list[str],
    ids: list[str],
    out_dir: str | Path,
    *,
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda",
    batch: int = 32,
    shard: int = SHARD,
    resume: bool = True,
) -> Path:
    """Verilen yollarin embedding'lerini shard'lar halinde diske yazar.

    Doner: cikti klasoru. Icerik:
        shard_00000.npy / shard_00000.ids.json / ...
        meta.json  (model adi, boyut, toplam sayi)
    """
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = len(paths)
    n_shards = (n + shard - 1) // shard
    todo = []
    for si in range(n_shards):
        npy = out / f"shard_{si:05d}.npy"
        idj = out / f"shard_{si:05d}.ids.json"
        if resume and npy.exists() and idj.exists():
            continue
        todo.append(si)

    if not todo:
        print(f"Tum {n_shards} parca zaten var -- cikarma atlandi.")
        _write_meta(out, model_name, n, n_shards)
        return out

    print(f"{n} goruntu / {n_shards} parca | {len(todo)} parca uretilecek")
    model, proc, device = load_clip(model_name, device)

    t0 = time.time()
    done_imgs = 0
    for si in todo:
        lo, hi = si * shard, min((si + 1) * shard, n)
        chunk_paths, chunk_ids = paths[lo:hi], ids[lo:hi]

        vecs = np.zeros((len(chunk_paths), EMBED_DIM), dtype=np.float32)
        kept_ids: list[str] = []
        row = 0

        for b0 in range(0, len(chunk_paths), batch):
            imgs, ok_ids = [], []
            for p, i in zip(chunk_paths[b0:b0 + batch], chunk_ids[b0:b0 + batch]):
                rgb = _read_rgb(p)
                if rgb is None:
                    print(f"  ATLANDI (okunamadi): {p}")
                    continue
                imgs.append(rgb)
                ok_ids.append(i)
            if not imgs:
                continue

            inputs = proc(images=imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = _image_embeds(model, inputs)
            vecs[row:row + len(imgs)] = feats.detach().cpu().numpy().astype(np.float32)
            kept_ids.extend(ok_ids)
            row += len(imgs)
            done_imgs += len(imgs)

            if done_imgs % (batch * 10) < batch:
                hz = done_imgs / max(1e-9, time.time() - t0)
                kalan = (n - lo - row) / max(1e-9, hz)
                print(f"  {lo + row}/{n}  ({hz:.1f} goruntu/sn, ~{kalan/60:.0f} dk kaldi)")

        # Okunamayan dosyalar varsa fazla satirlari kirp
        vecs = vecs[:row]
        np.save(out / f"shard_{si:05d}.npy", vecs)
        (out / f"shard_{si:05d}.ids.json").write_text(
            json.dumps(kept_ids, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  parca {si} yazildi: {vecs.shape}")

    _write_meta(out, model_name, n, n_shards)
    print(f"Tamam: {out}  ({(time.time()-t0)/60:.1f} dk)")
    return out


def _write_meta(out: Path, model_name: str, n: int, n_shards: int) -> None:
    (out / "meta.json").write_text(
        json.dumps(
            {"model": model_name, "dim": EMBED_DIM, "n_requested": n, "n_shards": n_shards},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Okuma
# ---------------------------------------------------------------------------

def load_embeddings(cache_dir: str | Path | list) -> tuple[np.ndarray, list[str]]:
    """Tum parcalari birlestirip (X, ids) doner.

    Birden fazla klasor verilebilir. Bu, isin bolunerek yapildigi durumlar
    icindir: orn. buyuk katmanlar Kaggle T4'te, geride kalan kucuk bir katman
    yerelde CPU'da cikarilir. Ayni image_id iki klasorde de varsa ILKI
    kazanir ve tekrar sayilmaz.
    """
    dirs = [Path(d) for d in (cache_dir if isinstance(cache_dir, (list, tuple)) else [cache_dir])]

    Xs: list[np.ndarray] = []
    ids: list[str] = []
    seen: set[str] = set()
    bulunan = 0

    for cache in dirs:
        shards = sorted(cache.glob("shard_*.npy"))
        if not shards:
            print(f"  UYARI: {cache} altinda parca yok, atlandi.")
            continue
        bulunan += len(shards)
        for npy in shards:
            idj = Path(str(npy)[:-4] + ".ids.json")
            if not idj.exists():
                raise FileNotFoundError(f"{npy} yaninda .ids.json yok -- hizalama yapilamaz.")
            arr = np.load(npy)
            shard_ids = json.loads(idj.read_text(encoding="utf-8"))
            if len(shard_ids) != len(arr):
                raise RuntimeError(
                    f"{npy.name}: {len(arr)} embedding ama {len(shard_ids)} id."
                )
            # Klasorler arasi cakismalari ele
            yeni = [k for k, i in enumerate(shard_ids) if i not in seen]
            if len(yeni) < len(shard_ids):
                print(f"  {npy.name}: {len(shard_ids) - len(yeni)} tekrar eden id atlandi.")
            if not yeni:
                continue
            Xs.append(arr[yeni])
            for k in yeni:
                ids.append(shard_ids[k])
                seen.add(shard_ids[k])

    if not Xs:
        raise FileNotFoundError(
            f"Hicbir parca bulunamadi: {[str(d) for d in dirs]}. "
            "Once `python -m src.features.clip_embed` calistir."
        )
    X = np.vstack(Xs)
    if len(ids) != len(X):
        raise RuntimeError(f"Hizalama bozuk: {len(X)} embedding ama {len(ids)} id.")
    print(f"  {bulunan} parca -> {len(X)} embedding ({len(dirs)} klasorden)")
    return X, ids


def align_to_manifest(X: np.ndarray, ids: list[str], df) -> tuple[np.ndarray, "object"]:
    """Embedding'leri manifest satir sirasina hizalar.

    Satir sirasina guvenmek yerine image_id uzerinden eslestirir; manifest
    yeniden uretildiginde sira degisebilir. Embedding'i olmayan satirlar
    (okunamamis dosyalar) DUSURULUR ve sayisi bildirilir."""
    pos = {i: k for k, i in enumerate(ids)}
    mask = df["image_id"].astype(str).map(lambda i: i in pos)
    kayip = int((~mask).sum())
    if kayip:
        print(f"UYARI: {kayip} manifest satirinin embedding'i yok, dusuruldu.")
    sub = df[mask].copy()
    rows = sub["image_id"].astype(str).map(pos).to_numpy()
    return X[rows], sub


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.manifest import load_manifest

    ap = argparse.ArgumentParser(description="CLIP ViT-L/14 embedding cikar ve cache'le")
    ap.add_argument("--manifest", default="data/processed/manifest_v2_laundered.parquet")
    ap.add_argument("--out", default=str(DEFAULT_CACHE))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--split", default=None, help="Sadece bu split (train/val/test)")
    ap.add_argument("--profile", default=None, help="Sadece bu laundering profili")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--shard", type=int, default=SHARD)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    df = load_manifest(a.manifest)
    if a.split:
        df = df[df["split"].astype(str) == a.split]
    if a.profile:
        df = df[df["launder_profile"].astype(str) == a.profile]
    if len(df) == 0:
        raise SystemExit("Filtreden sonra hic satir kalmadi.")

    # Deterministik sira: yeniden calistirmada parca sinirlari kaysin istemiyoruz
    df = df.sort_values("image_id").reset_index(drop=True)

    print(f"Manifest: {a.manifest}  ({len(df)} satir)")
    print(f"Model   : {a.model}")
    print(f"Cache   : {a.out}")

    embed_paths(
        df["path"].astype(str).tolist(),
        df["image_id"].astype(str).tolist(),
        a.out,
        model_name=a.model,
        device=a.device,
        batch=a.batch,
        shard=a.shard,
        resume=not a.no_resume,
    )


if __name__ == "__main__":
    _cli()
