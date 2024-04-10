
from PIL import Image

import torch
from torch.nn import functional as F

from typing import List, Optional, Union

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers import DiffusionPipeline

from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler
from diffusers.utils import logging

# TODO: remove and import from diffusers.utils when the new version of diffusers is released

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import numpy as np
from torchvision import transforms

# 1. Load the autoencoder model which will be used to decode the latents into image space.
def SD_pretrained_load(SD_MODEL_NAME,CLIP_MODEL_NAME):
    vae = AutoencoderKL.from_pretrained(
    SD_MODEL_NAME, subfolder='vae', use_auth_token=True).to(device)

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
    text_encoder = CLIPTextModel.from_pretrained(CLIP_MODEL_NAME).to(device)

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(
        SD_MODEL_NAME, subfolder='unet', use_auth_token=True).to(device)

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

    logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
    return vae,tokenizer,text_encoder,unet,scheduler


def preprocess(image,PIL_INTERPOLATION):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class StableDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for imagic image editing.
    See paper here: https://arxiv.org/pdf/2210.09276.pdf
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offsensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, LMSDiscreteScheduler]):
        super().__init__()


    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.
        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.
        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    @torch.no_grad()
    def Imagic_base_pipeline(self,
        cond_embeddings,
        input_latents,
        seed:int=0,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: float = 7.5,
        ):

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens = [""]
            max_length = self.tokenizer.model_max_length
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.view(1, seq_len, -1)

            cond_embeddings = torch.cat([uncond_embeddings, cond_embeddings])


        torch.manual_seed(seed)
        self.scheduler.set_timesteps(num_inference_steps)
        noise = torch.randn_like(input_latents)
        bsz = input_latents.shape[0]
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        if self.scheduler is LMSDiscreteScheduler:
            noisy_latents = input_latents * self.scheduler.sigmas[0]

        loss_avg = AverageMeter()
        for i, t in tqdm(enumerate((timesteps_tensor))):
            # expand the latents if we are doing classifier free guidance
            noisy_latents = self.scheduler.add_noise(input_latents, noise, t)
            latent_model_input = torch.cat([noisy_latents] * 2) if do_classifier_free_guidance else noisy_latents
            if self.scheduler is LMSDiscreteScheduler:
                sigma = self.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)
            # print(latent_model_input.size())
            # print(text_embeddings.size())
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=cond_embeddings)['sample']
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            loss_avg.update(loss.detach_(), bsz)
            latents = self.scheduler.step(noise_pred, t, noisy_latents)['prev_sample']
        return loss_avg, latents

    def generateImage(self,
        cond_embeddings,
        seed:int=0,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        output_type: Optional[str] = "pil",
        guidance_scale: float = 7.5,
    ):

        # get the initial random noise unless the user supplied it
        latents_shape = (1, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = cond_embeddings.dtype
        torch.manual_seed(seed)
        input_latents = torch.randn(latents_shape, dtype=latents_dtype.to(self.device))

        _,latents = self.Imagic_base_pipeline(cond_embeddings,input_latents,seed,height,width,num_inference_steps,guidance_scale)
        latents = 1 / 0.18215 * latents
        images = self.vae.decode(latents)

        images = (images.sample / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        images = images.detach().cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            # image = self.numpy_to_pil(image)
            images = (images * 255).round().astype('uint8')
            images = [Image.fromarray(image) for image in images]

        return images


    @torch.no_grad()
    def conditioned_diffusion_loss(self,
        cond_embeddings: torch.Tensor,
        image,
        seed: int = 0,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        resolution: Optional[int] = 512,
        center_crop: bool = False,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: float = 7.5,

    ):

        #encode classfier image using VAE
        # input_image = image.convert("RGB")
        input_image = image
        image_transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        input_image = image_transforms(input_image)
        input_image = input_image[None].to(device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            input_latents = self.vae.encode(input_image).latent_dist.sample()
            input_latents = 0.18215 * input_latents

        loss_avg,_ = self.Imagic_base_pipeline(cond_embeddings,input_latents,seed,height,width,num_inference_steps,guidance_scale)


        return  loss_avg






