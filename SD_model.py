
import torch
from typing import List, Optional, Union
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler
from diffusers.utils import logging

from transformers import CLIPTextModel, CLIPTokenizer

import os
import gc
import data_upload
import helper_functions
# import SD_pipeline
import new_SD_pipline


def SD_pretrained_load(SD_MODEL_NAME,CLIP_MODEL_NAME,device,imagic_trained =False):
    if imagic_trained:
        vae_path = os.path.join(SD_MODEL_NAME,'vae')
        tokenizer_path = os.path.join(SD_MODEL_NAME,'tokenizer')
        text_encoder_path = os.path.join(SD_MODEL_NAME,'text_encoder')
        unet_path =   os.path.join(SD_MODEL_NAME,'unet')
    else:
        vae_path = unet_path = SD_MODEL_NAME
        tokenizer_path = text_encoder_path = CLIP_MODEL_NAME
    vae = AutoencoderKL.from_pretrained(
    vae_path, subfolder='vae', token=True).to(device)

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path).to(device)

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(
        unet_path, subfolder='unet', token=True).to(device)

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

    logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
    return vae,text_encoder,tokenizer,unet,scheduler

def all_generator(all_files,output_folder,imagic_pretrained_path,Imagic_pipe,SD_model_name, CLIP_model_name, device,
                  seed_range,alpha_range,guidance_scale_range,cat_embeds=None,height: Optional[int] = 512,
                  width: Optional[int] = 512,num_inference_steps: Optional[int] = 50):
    for embeds in all_files:
        if cat_embeds is not None:
            cat_embeddings = cat_embeds[embeds][0] + cat_embeds[embeds][1] / cat_embeds[embeds][2]
            Imagic_pipe=False
        else:
            cat_embeddings =None

        if Imagic_pipe:
            pipe_name = 'Imagic_pipeline'
            SD_pretrained_model = None
        else:
            pipe_name = 'SD_pipeline'
            SD_pretrained_model = SD_pretrained_load(SD_model_name, CLIP_model_name, device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        imagic_parameters = data_upload.upload_single_imagic_params(imagic_pretrained_path, embeds, CLIP_model_name,
                                                                    device, Imagic_pipe)
        for seed in seed_range:
            output_dir = os.path.join(output_folder, pipe_name, "seed_{}".format(str(seed)))
            os.makedirs(output_dir, exist_ok=True)
            for alpha in alpha_range:
                for guidance_scale in guidance_scale_range:
                    image_checkpoint, image_path, embeds_file = helper_functions.generated_image_checkpoint(output_dir,
                                                                                                            embeds,
                                                                                                            alpha,
                                                                                                            guidance_scale)
                    print(image_checkpoint, image_path, embeds_file)
                    cat_path, image_name = image_path.rsplit("/", 1)
                    os.makedirs(cat_path, exist_ok=True)
                    if image_checkpoint:
                        continue
                    else:
                        image_generator(cat_path,
                                                 imagic_parameters,
                                                 image_name,
                                                 cat_embeddings,
                                                 SD_pretrained_model,
                                                 alpha,
                                                 seed=seed,
                                                 height=height,
                                                 width=width,
                                                 num_inference_steps=num_inference_steps,
                                                 guidance_scale=guidance_scale)


def image_generator(output_folder,imagic_parameters,image_name,cat_embeds=None,SD_pretrained_models=None,alpha = 0,
    seed: int = 0,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: Optional[int] = 50,
    guidance_scale: float = 7.5):


    if SD_pretrained_models is not None:
        pipeline = new_SD_pipline.StableDiffusionPipeline(*SD_pretrained_models)
        _, target_embeddings, optimized_embeddings = imagic_parameters
    else:
        pipeline, target_embeddings, optimized_embeddings = imagic_parameters
    print("generating image {}".format(image_name))
    if cat_embeds is not None:
        embeddings = cat_embeds
    else:
        embeddings = alpha * target_embeddings + (1 - alpha) * optimized_embeddings
    with torch.autocast("cuda"), torch.inference_mode():
        images = pipeline.generateImage(
                cond_embeddings = embeddings,
                seed=seed,
                height = height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
    image =images[0]
    image.save(os.path.join(output_folder, image_name))

def conditioned_classifier(parameters,test_image,
    seed: int = 0,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    resolution: Optional[int] = 512,
    num_inference_steps: Optional[int] = 50,
    guidance_scale: float = 7.5):

    SD_loss = {}

    for embeds_name,params in parameters.items():
        pipeline, embeddings = params
        print(len(params))
        print(type(pipeline))
        with torch.autocast("cuda"), torch.inference_mode():
            # loss_avg = pipeline.conditioned_diffusion_loss(
            #     cond_embeddings=embeddings,
            #     image=test_image.convert('RGB'),
            #     seed=seed,
            #     height=height,
            #     width=width,
            #     resolution=resolution,
            #     num_inference_steps=num_inference_steps,
            #     guidance_scale=guidance_scale
            #
            loss_avg = pipeline.diffusionloss_IM_IM(
                image_ID_embeddings=embeddings,
                image=test_image.convert('RGB'),
                seed=seed,
                height=height,
                width=width,
                resolution=resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )

        SD_loss[embeds_name] = (loss_avg.avg.item())

    return SD_loss


def all_embeds_conditioned_classifier(imagic_pretrained_path,csv_folder,SD_model_name,CLIP_model_name,device,image_list,category_class=False,
                                      Imagic_pipe=False,alpha = 0,seed: int = 0,height: Optional[int] = 512,width: Optional[int] = 512,
                                      resolution: Optional[int] = 512,num_inference_steps: Optional[int] = 50,guidance_scale: float = 7.5):

    if Imagic_pipe:
        pipe_name = 'Imagic_pipeline'
        SD_pretrained_model = None
        all_files = set(os.listdir(imagic_pretrained_path))
    else:
        pipe_name = 'SD_pipeline'
        SD_pretrained_model = SD_pretrained_load(SD_model_name, CLIP_model_name, device)
        print(type(SD_pretrained_model))
        if category_class:
            pipe_name = 'SD_embeds_cat_avg'
            cat_files = data_upload.upload_cat_embeds(imagic_pretrained_path, CLIP_model_name, device,Imagic_pipe, SD_pretrained_model)
            all_files = list(cat_files.keys())
        else:
            all_files = set(os.listdir(imagic_pretrained_path))
    for file in all_files:
        if category_class:
            embeds_files = {file:cat_files[file]}
            # print(embeds_files.keys())
            # print(type(cat_files[file][0]))
            # print(type(embeds_files[file][0]))
        else:

            embeds_files = data_upload.upload_embeds(imagic_pretrained_path,file, CLIP_model_name,alpha, device,Imagic_pipe,
                                             SD_pretrained_model)

        csv_dir = os.path.join(csv_folder, pipe_name)
        os.makedirs(csv_dir, exist_ok=True)

        for image_name, image, _ in image_list:
            cls = image_name.rsplit("_", 1)[0]
            image_flag, df_sd, csv_file_path = helper_functions.csv_checkpoint(csv_dir, cls, image_name,file)
            if not image_flag:
                SD_loss =conditioned_classifier(embeds_files, image,seed, height, width,resolution, num_inference_steps,guidance_scale)
                helper_functions.save_to_csv(SD_loss,df_sd,image_name,csv_file_path)








