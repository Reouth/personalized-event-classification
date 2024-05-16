import clip
import torch
import os
import data_upload
class CLIP_pipline():
       def __init__(self,device,model_name):
              self.model_CLIP, self.preprocess_CLIP = clip.load(model_name, device)
              self.device = device

       def image_to_embedding(self,
               image):
              CLIP_image = self.preprocess_CLIP(image).unsqueeze(0).to(self.device)
              with torch.no_grad():
                     image_features = self.model_CLIP.encode_image(CLIP_image)
              image_features /= image_features.norm(dim=-1, keepdim=True)
              return image_features

       def images_to_embeddings(self, path):
              clip_embeddings= {}
              images = data_upload.upload_images(path)
              for name,img in images:
                     clip_embeddings[name] = self.image_to_embedding(img)
              return clip_embeddings

