

from PIL import Image
import torch
from torch.nn import functional as F
from typing import List, Optional, Union

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers import DiffusionPipeline

from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import numpy as np
from torchvision import transforms

# logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

#
# def preprocess(image):
#     w, h = image.size
#     w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
#     image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
#     image = np.array(image).astype(np.float32) / 255.0
#     image = image[None].transpose(0, 3, 1, 2)
#     image = torch.from_numpy(image)
#     return 2.0 * image - 1.0

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
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler
        )

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
    def generateImage(self,
        text_embeddings,
        seed:int=0,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        **kwargs
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        # if optimized_embeddings is None:
        #     raise ValueError("Please run the pipe.train() before trying to generate an image.")
        # if target_embeddings is None:
        #     raise ValueError("Please run the pipe.train() before trying to generate an image.")

        # # text_embeddings = alpha * target_embeddings + (1 - alpha) * optimized_embeddings

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.

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

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.view(1, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_shape = (1, self.unet.in_channels, height // 8, width // 8)
        print(latents_shape)
        latents_dtype = text_embeddings.dtype
        torch.manual_seed(seed)
        latents = torch.randn(latents_shape, dtype=latents_dtype)
        latents = latents.to(self.device)
        self.scheduler.set_timesteps(num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        if self.scheduler is LMSDiscreteScheduler:
          latents = latents * self.scheduler.sigmas[0]


        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        # accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # extra_step_kwargs = {}
        # if accepts_eta:
        #     extra_step_kwargs["eta"] = eta
        for i, t in tqdm(enumerate((timesteps_tensor))):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            if self.scheduler is LMSDiscreteScheduler:
              sigma = self.scheduler.sigmas[i]
              latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            # print(latent_model_input.size())
            # print(text_embeddings.size())
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        latents = 1 / 0.18215 * latents
        images = self.vae.decode(latents)

        images = (images.sample / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        images = images.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        # if self.safety_checker is not None:
        #     safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(
        #         self.device
        #     )
        #     image, has_nsfw_concept = self.safety_checker(
        #         images=image, clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype)
        #     )
        # else:
        #     has_nsfw_concept = None

        if output_type == "pil":
            # image = self.numpy_to_pil(image)
            images = (images * 255).round().astype('uint8')
            images = [Image.fromarray(image) for image in images]

        return images


    @torch.no_grad()
    def diffusionloss_IM_text(self,
        text_embeddings: torch.Tensor,
        input_image,
        seed: int = 0,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        resolution: Optional[int] = 512,
        center_crop: bool = False,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: float = 7.5,
        **kwargs
    ):

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        #encode classfier image using VAE
        # input_image = image.convert("RGB")

        image_transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        init_image = image_transforms(input_image)
        init_image = init_image[None].to(device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            init_latents = self.vae.encode(init_image).latent_dist.sample()
            init_latents = 0.18215 * init_latents


        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
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

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.view(1, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents_shape = (1, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype


        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        torch.manual_seed(seed)
        noise = torch.randn_like(init_latents)
        bsz = init_latents.shape[0]
        #calculate diffusion loss for noised image at each timestep given text_embeddings
        loss_avg = AverageMeter()
        for i, t in tqdm(enumerate((timesteps_tensor))):

  # expand the latents if we are doing classifier free guidance
            noisy_latents = self.scheduler.add_noise(init_latents, noise, t)
            noisy_latents = torch.cat([noisy_latents] * 2) if do_classifier_free_guidance else noisy_latents
            # perform guidance
            noise_pred = self.unet(noisy_latents, t, text_embeddings)['sample']
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)


            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            loss_avg.update(loss.detach_(), bsz)
            # print("loss:" + str(loss.item()))
            # print("avg loss:" + str(loss_avg.avg.item()))
            # print(i)
            # if not step % args.log_interval:
            #     logs = {"loss": loss_avg.avg.item()}


        return  loss_avg

    @torch.no_grad()
    def diffusionloss_IM_IM(self,
        image_ID_embeddings: torch.Tensor,
        image,
        seed: int = 0,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        resolution: Optional[int] = 512,
        center_crop: bool = False,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: float = 7.5,
        **kwargs
    ):

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # #encode classfier image using VAE
        # # input_image = image.convert("RGB")
        input_image = image
        image_transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        init_image = image_transforms(input_image)
        input_image = image
        init_image = init_image[None].to(device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            init_latents = self.vae.encode(init_image).latent_dist.sample()
            init_latents = 0.18215 * init_latents

        # # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # # corresponds to doing no classifier free guidance.
        # do_classifier_free_guidance = guidance_scale > 1.0
        # # get unconditional embeddings for classifier free guidance
        # if do_classifier_free_guidance:
        #     uncond_tokens = [""]
        #     max_length = self.tokenizer.model_max_length
        #     uncond_input = self.tokenizer(
        #         uncond_tokens,
        #         padding="max_length",
        #         max_length=max_length,
        #         truncation=True,
        #         return_tensors="pt",
        #     )
        #     uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        #     # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        #     seq_len = uncond_embeddings.shape[1]
        #     uncond_embeddings = uncond_embeddings.view(1, seq_len, -1)

        #     # For classifier free guidance, we need to do two forward passes.
        #     # Here we concatenate the unconditional and text embeddings into a single batch
        #     # to avoid doing two forward passes
        #     text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents_shape = (1, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = image_ID_embeddings.dtype


        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        torch.manual_seed(seed)
        noise = torch.randn_like(init_latents)
        bsz = init_latents.shape[0]
        #calculate diffusion loss for noised image at each timestep given text_embeddings
        loss_avg = AverageMeter()
        for i, t in tqdm(enumerate((timesteps_tensor))):

  # expand the latents if we are doing classifier free guidance
            noisy_latents = self.scheduler.add_noise(init_latents, noise, t)
            # noisy_latents = torch.cat([noisy_latents] * 2) if do_classifier_free_guidance else noisy_latents
            # perform guidance
            noise_pred = self.unet(noisy_latents, t, image_ID_embeddings)['sample']
            # if do_classifier_free_guidance:
            #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)


            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            loss_avg.update(loss.detach_(), bsz)
            # print("loss:" + str(loss.item()))
            # print("avg loss:" + str(loss_avg.avg.item()))
            # print(i)
            # if not step % args.log_interval:
            #     logs = {"loss": loss_avg.avg.item()}


        return  loss_avg





