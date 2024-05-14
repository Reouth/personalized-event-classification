import torch
import SD_imagic_train
from diffusers import DiffusionPipeline
from PIL import Image

import torch
from torch.nn import functional as F
import accelerate
from typing import List, Optional, Union

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers import DiffusionPipeline

from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler
from diffusers.utils import logging
import os
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import numpy as np
from torchvision import transforms
class embedding_generator(DiffusionPipeline):

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

    def text_to_embedding(self,target_text):

        text_ids = self.tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        text_ids = text_ids.to(device=self.device)
        with torch.inference_mode():
            target_embeddings = self.text_encoder(text_ids)[0].to(device=self.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        target_embeddings = target_embeddings.float().to(device=self.device)
        return target_embeddings

    def load_imagic_embedding(self,text,image_path,folder_path,pretrained_model_name_or_path):

        img_name = image_path.rsplit('.', 1)[0].replace(" ", "_")
        text_path = text.replace(" ", "_")
        OUTPUT_DIR = os.path.join(folder_path, img_name + '*' + text_path)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        target_embed_path = os.path.join(OUTPUT_DIR, "target_embeddings.pt")
        optimized_embed_path = os.path.join(OUTPUT_DIR, "optimized_embeddings.pt")

        if os.path.isfile(target_embed_path) and os.path.isfile(optimized_embed_path):
            print("embedding loaded from folder {}".format(OUTPUT_DIR))

        else:
            print("embedding does't exist, generating embedding in folder {}".format(OUTPUT_DIR))
            SD_imagic_train.imagic(pretrained_model_name_or_path,
                                OUTPUT_DIR,
                                image_path,
                                text,
                                seed=3434554,
                                resolution=514,
                                mixed_precision="fp16",
                                use_8bit_adam=True,
                                gradient_accumulation_steps=1,
                                emb_learning_rate=1e-3,
                                learning_rate=2e-6,
                                emb_train_steps=2000,
                                max_train_steps=4000)
        return torch.load(target_embed_path).to(self.device), torch.load(optimized_embed_path).to(self.device)


