"""The eight base text-to-image models used in the paper and how we sample them.

Each `load_<model>` function loads a pipeline onto `device` and returns a
`sample(prompt, seed) -> PIL.Image` function. Every model is sampled with
50 denoising steps and a guidance scale of 10 (DeepFloyd IF uses the diffusers
default number of steps for each stage).

To add a model, write a loader and register it in `MODELS`.
"""

import torch
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    FluxPipeline,
    HunyuanDiTPipeline,
    KolorsPipeline,
    PixArtSigmaPipeline,
    SanaPipeline,
    StableDiffusionPipeline,
)
from PIL import Image

NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 10


def _sampler(pipeline, resolution, **pipeline_kwargs):
    """Wrap a standard diffusers pipeline into a `sample(prompt, seed)` function."""
    pipeline.set_progress_bar_config(disable=True)

    def sample(prompt, seed):
        return pipeline(
            prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            num_images_per_prompt=1,
            width=resolution,
            height=resolution,
            generator=torch.manual_seed(seed),
            **pipeline_kwargs,
        ).images[0]

    return sample


def load_sd21(checkpoint, device):
    pipeline = StableDiffusionPipeline.from_pretrained(
        checkpoint, torch_dtype=torch.float16, safety_checker=None
    )
    return _sampler(pipeline.to(device), resolution=768)


def load_sdxl(checkpoint, device):
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipeline = DiffusionPipeline.from_pretrained(
        checkpoint, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    return _sampler(pipeline.to(device), resolution=1024)


def load_flux(checkpoint, device):
    pipeline = FluxPipeline.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
    return _sampler(pipeline.to(device), resolution=1024, max_sequence_length=512)


def load_hunyuan(checkpoint, device):
    pipeline = HunyuanDiTPipeline.from_pretrained(checkpoint, torch_dtype=torch.float16)
    return _sampler(pipeline.to(device), resolution=1024)


def load_kolors(checkpoint, device):
    pipeline = KolorsPipeline.from_pretrained(checkpoint, torch_dtype=torch.float16, variant="fp16")
    return _sampler(pipeline.to(device), resolution=1024)


def load_pixartsigma(checkpoint, device):
    pipeline = PixArtSigmaPipeline.from_pretrained(
        checkpoint, torch_dtype=torch.float16, use_safetensors=True
    )
    return _sampler(pipeline.to(device), resolution=1024)


def load_sana(checkpoint, device):
    pipeline = SanaPipeline.from_pretrained(checkpoint, variant="fp16", torch_dtype=torch.float16)
    pipeline = pipeline.to(device)
    pipeline.vae.to(torch.bfloat16)
    pipeline.text_encoder.to(torch.bfloat16)
    return _sampler(pipeline, resolution=1024)


def load_deepfloyd(checkpoint, device):
    """DeepFloyd IF: stage I (64px) followed by `checkpoint` as stage II (256px)."""
    stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", torch_dtype=torch.float16)
    stage_1 = stage_1.to(device)
    stage_1.watermarker = None
    stage_2 = DiffusionPipeline.from_pretrained(
        checkpoint, text_encoder=None, torch_dtype=torch.float16
    )
    stage_2 = stage_2.to(device)
    stage_1.set_progress_bar_config(disable=True)
    stage_2.set_progress_bar_config(disable=True)

    def sample(prompt, seed):
        generator = torch.Generator(device=device).manual_seed(seed)
        prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)
        image = stage_1(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
            output_type="pt",
        ).images
        image = stage_2(
            image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
            output_type="pt",
        ).images[0]
        # [-1, 1] CHW tensor -> uint8 HWC image
        image = (image / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
        return Image.fromarray((image * 255).round().astype("uint8"))

    return sample


# name -> (loader, default Hugging Face checkpoint)
MODELS = {
    "sd21": (load_sd21, "stabilityai/stable-diffusion-2-1"),
    "sdxl": (load_sdxl, "stabilityai/stable-diffusion-xl-base-1.0"),
    "deepfloyd": (load_deepfloyd, "DeepFloyd/IF-II-L-v1.0"),
    "flux": (load_flux, "black-forest-labs/FLUX.1-dev"),
    "hunyuan": (load_hunyuan, "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"),
    "kolors": (load_kolors, "Kwai-Kolors/Kolors-diffusers"),
    "pixartsigma": (load_pixartsigma, "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"),
    "sana": (load_sana, "Efficient-Large-Model/Sana_1600M_1024px_diffusers"),
}


def load_sampler(model, device, checkpoint=None):
    """Return a `sample(prompt, seed)` function for `model`.

    `checkpoint` overrides the default weights, e.g. a local fine-tuned SD2.1.
    """
    loader, default_checkpoint = MODELS[model]
    return loader(checkpoint or default_checkpoint, device)
