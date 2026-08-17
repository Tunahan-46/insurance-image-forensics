"""
Difuzyon hatti yukleyicileri (tek nokta).

NEDEN AYRI DOSYA
----------------
fully_synthetic / inpaint_add / inpaint_remove ayni model yukleme, dtype,
scheduler ve VRAM yonetimi mantigini paylasir. Bunu uc yere kopyalarsak
"Colab'da calisiyor ama yerelde patliyor" sinifi hatalar uc kat artar.

KRITIK: torch/diffusers import'lari FONKSIYON ICINDE yapilir. Bu sayede
`src.data.generators` paketi GPU'suz bir makinede (ornegin CI'da veya
classic_manip.py calistirirken) sorunsuz import edilir.

MODEL SECIMI (plan 4.2 ve Hafta 2 split kurali 2)
-------------------------------------------------
    sd15         -> train/val   (hizli, zayif; cesitlilik icin)
    sdxl         -> train/val   (kaliteli, ana uretici)
    flux_schnell -> SADECE TEST (gorulmemis generator deneyi, E6)

flux_schnell'i train'e sokmak "unseen generator" deneyini gecersiz kilar.
Bu kisit `MODEL_REGISTRY` icinde `test_only` bayragiyla kodlanmistir ve
scripts/build_manifest_v2.py bunu makine olarak dogrular.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ModelKey = Literal["sd15", "sdxl", "flux_schnell"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    inpaint_repo: str | None
    default_steps: int
    default_guidance: float
    native_size: int
    test_only: bool
    note: str
    # Ana depo 404 verirse sirayla denenecek aynalar. RunwayML 2024'te
    # SD 1.5 depolarini Hugging Face'ten KALDIRDI; eski adresleri
    # kullanan her kod bir gun aniden calismaz oldu. Tek bir depo adina
    # bagli kalmak, gece boyu surecek bir uretimi sabaha bos elle
    # uyanmaya cevirir.
    repo_fallbacks: tuple[str, ...] = ()
    inpaint_fallbacks: tuple[str, ...] = ()
    # Yaklasik agirlik boyutu (GB). Colab'da indirme suresini ve VRAM
    # riskini onceden tahmin etmek icin.
    approx_gb: float = 0.0


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "sd15": ModelSpec(
        key="sd15",
        repo="stable-diffusion-v1-5/stable-diffusion-v1-5",
        repo_fallbacks=(
            "runwayml/stable-diffusion-v1-5",  # kaldirildi, yine de denenir
            "botp/stable-diffusion-v1-5",
        ),
        inpaint_repo="stable-diffusion-v1-5/stable-diffusion-inpainting",
        inpaint_fallbacks=(
            "runwayml/stable-diffusion-inpainting",
            "botp/stable-diffusion-v1-5-inpainting",
        ),
        default_steps=30,
        default_guidance=7.5,
        native_size=512,
        test_only=False,
        approx_gb=4.0,
        note="Hizli ve zayif. Cesitlilik icin degerli: dedektor sadece SDXL "
        "gorurse zayif ureticilere genellemeyi ogrenemez. T4'te rahat calisir.",
    ),
    "sdxl": ModelSpec(
        key="sdxl",
        repo="stabilityai/stable-diffusion-xl-base-1.0",
        inpaint_repo="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        # Son care: SDXL base checkpoint'i de inpaint hattina yuklenebilir
        # (AutoPipelineForInpainting bunu destekler). Kalitesi inpaint'e
        # ozel checkpoint'ten dusuktur ama M1 katmanini SIFIR ornekle
        # birakmaktan iyidir. Kullanilirsa gen_log'daki `repo` alanindan
        # gorulur ve dataset_card'a not dusulmelidir.
        inpaint_fallbacks=("stabilityai/stable-diffusion-xl-base-1.0",),
        default_steps=30,
        default_guidance=6.0,
        native_size=1024,
        test_only=False,
        approx_gb=7.0,
        note="Ana uretici. Inpainting kalitesi M1 senaryosunun zorlugunu belirler. "
        "T4'te model CPU offload ile calisir.",
    ),
    "flux_schnell": ModelSpec(
        key="flux_schnell",
        repo="black-forest-labs/FLUX.1-schnell",
        inpaint_repo=None,  # resmi inpaint varyanti yok; FLUX yalnizca S katmani
        default_steps=4,  # schnell distilled: 4 adim yeterli
        default_guidance=0.0,  # schnell guidance kullanmaz
        native_size=1024,
        test_only=True,
        approx_gb=24.0,
        note="SADECE TEST. 12 milyar parametre, ~24 GB. UCRETSIZ COLAB T4'TE "
        "RISKLI: 16 GB VRAM ve Turing mimarisi bf16 desteklemiyor. Once "
        "notebooks/W2_colab_smoke_test ile DENE; gecmezse 'gorulmemis "
        "uretici' deneyini SD1.5-train / SDXL-test kurgusuyla yap.",
    ),
}

TEST_ONLY_GENERATORS: set[str] = {k for k, v in MODEL_REGISTRY.items() if v.test_only}

# Cikti cozunurluk havuzu (plan 4.2: "cesitli cozunurluk ve en-boy orani").
# TEK cozunurlukta uretmek plan 4.5 Tuzak 2'yi (model cozunurlugu ogrenir)
# dogrudan tetikler. W1 Bulgu 3 zaten bunun canli ornegi.
RESOLUTION_POOL: dict[str, list[tuple[int, int]]] = {
    "sd15": [(512, 512), (512, 640), (640, 512), (768, 512), (512, 768)],
    "sdxl": [(1024, 1024), (1024, 768), (768, 1024), (1152, 896), (896, 1152)],
    "flux_schnell": [(1024, 1024), (1024, 768), (768, 1024)],
}


def resolve_dtype(device: str) -> Any:
    import torch

    return torch.float16 if device == "cuda" else torch.float32


def _try_repos(loader, repos: list[str], **kwargs):
    """Depolari sirayla dener, ilk basarili olani doner.

    IKI EKSENDE YEDEKLILIK
    ----------------------
    1. Depo adresi: RunwayML SD 1.5 depolarini kaldirdi; tek adrese bagli
       kalmak gece boyu surecek bir uretimi sabaha bos elle uyanmaya cevirir.

    2. `variant="fp16"`: HER depo fp16 agirlik dosyasi barindirmaz. Barindirmayan
       bir depoya variant gecirmek OSError verir ve hat komple olur -- ama
       ayni depo variant'siz MUKEMMEL calisir (fp32 dosyalari torch_dtype ile
       zaten fp16'ya cevrilir, sadece indirme boyutu buyur). Bu yuzden her
       depo icin once variant'li, sonra variant'siz deneniyor.

    Hepsi basarisiz olursa TOPLU bir hata mesaji verir -- hangi adreslerin
    ve hangi varyantlarin denendigini gosterir. Tek bir "404" mesajiyla bas
    basa kalmak, Colab'da saat kaybettirir.
    """
    errors = []
    has_variant = kwargs.get("variant") is not None

    for r in repos:
        attempts = [kwargs]
        if has_variant:
            fallback = {k: v for k, v in kwargs.items() if k != "variant"}
            attempts.append(fallback)

        for kw in attempts:
            tag = f"{r} (variant={kw.get('variant', 'yok')})" if has_variant else r
            try:
                print(f"  deneniyor: {tag}")
                return loader(r, **kw), r
            except Exception as e:  # noqa: BLE001 - hangi hata olursa olsun sonrakini dene
                errors.append(f"    {tag}: {type(e).__name__}: {str(e)[:180]}")

    raise RuntimeError(
        "Hicbir depo yuklenemedi. Denenenler:\n" + "\n".join(errors)
    )


def load_txt2img(model: str, device: str = "cuda", *, enable_cpu_offload: bool = True):
    """Metin->goruntu hatti yukler. Colab T4'te SDXL'i sigdirmak icin
    varsayilan olarak model CPU offload acilir."""
    spec = MODEL_REGISTRY[model]
    dtype = resolve_dtype(device)
    repos = [spec.repo, *spec.repo_fallbacks]

    if model == "flux_schnell":
        from diffusers import FluxPipeline

        pipe, used = _try_repos(FluxPipeline.from_pretrained, repos, torch_dtype=dtype)
    elif model == "sdxl":
        from diffusers import StableDiffusionXLPipeline

        pipe, used = _try_repos(
            StableDiffusionXLPipeline.from_pretrained, repos,
            torch_dtype=dtype, variant="fp16" if device == "cuda" else None,
            use_safetensors=True,
        )
    else:
        from diffusers import StableDiffusionPipeline

        pipe, used = _try_repos(
            StableDiffusionPipeline.from_pretrained, repos,
            torch_dtype=dtype, safety_checker=None,
        )

    print(f"  yuklendi: {used}")
    return _finalize(pipe, device, enable_cpu_offload)


def load_inpaint(model: str, device: str = "cuda", *, enable_cpu_offload: bool = True):
    """Inpainting hatti yukler. FLUX icin inpaint varyanti kayitli degildir
    ve bilincli olarak DESTEKLENMEZ -- test-only uretici manipulasyon
    katmanina girerse E6 deneyi kirilir."""
    spec = MODEL_REGISTRY[model]
    if spec.inpaint_repo is None:
        raise ValueError(
            f"{model} icin inpaint hatti yok. Bu kasitlidir: test-only "
            f"ureticiler yalnizca fully_synthetic katmaninda kullanilir."
        )
    dtype = resolve_dtype(device)
    repos = [spec.inpaint_repo, *spec.inpaint_fallbacks]

    if model == "sdxl":
        from diffusers import AutoPipelineForInpainting

        pipe, used = _try_repos(
            AutoPipelineForInpainting.from_pretrained, repos,
            torch_dtype=dtype, variant="fp16" if device == "cuda" else None,
        )
    else:
        from diffusers import StableDiffusionInpaintPipeline

        pipe, used = _try_repos(
            StableDiffusionInpaintPipeline.from_pretrained, repos,
            torch_dtype=dtype, safety_checker=None,
        )

    print(f"  yuklendi: {used}")
    return _finalize(pipe, device, enable_cpu_offload)


def _finalize(pipe, device: str, enable_cpu_offload: bool):
    pipe.set_progress_bar_config(disable=True)
    if device == "cuda" and enable_cpu_offload:
        # Colab T4 (16GB) SDXL inpaint icin sinirda. Sequential degil model
        # offload: sequential cok yavas, model offload hiz/VRAM dengesi iyi.
        pipe.enable_model_cpu_offload()
        try:
            pipe.enable_vae_slicing()
        except AttributeError:
            pass
    else:
        pipe.to(device)
    return pipe


def make_generator(seed: int, device: str = "cuda"):
    """Tekrarlanabilirlik icin sabit tohumlu torch generator."""
    import torch

    return torch.Generator(device="cpu" if device != "cuda" else "cuda").manual_seed(seed)


if __name__ == "__main__":
    print(f"{'model':<14} {'test_only':<10} {'adim':<6} {'guid':<6} {'native':<8} repo")
    print("-" * 92)
    for k, s in MODEL_REGISTRY.items():
        print(
            f"{k:<14} {s.test_only!s:<10} {s.default_steps:<6} "
            f"{s.default_guidance:<6} {s.native_size:<8} {s.repo}"
        )
    print(f"\nTest-only ureticiler: {TEST_ONLY_GENERATORS}")
    print("\nCozunurluk havuzlari:")
    for k, v in RESOLUTION_POOL.items():
        print(f"  {k}: {v}")
    assert MODEL_REGISTRY["flux_schnell"].inpaint_repo is None
    print("\npipelines.py sanity check OK (torch/diffusers import edilmedi)")
