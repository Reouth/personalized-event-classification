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
              return image_features.to(self.device)

       def images_to_embeddings(self, path):
              clip_embeddings= {}
              images = data_upload.upload_images(path)
              for name,img,_ in images:
                     clip_embeddings[name] = self.image_to_embedding(img)
              return clip_embeddings

       def conditioned_classifier(self,test_image,clip_embeddings):

              embeds_names = list(clip_embeddings.keys())
              CLIP_ID_embeds = torch.cat([clip_embeddings[embeds_name] for embeds_name in clip_embeddings]).to(
                     self.device)
              CLIP_ID_embeds /= CLIP_ID_embeds.norm(dim=-1, keepdim=True)

              test_embeddings = self.image_to_embedding(test_image)
              # Pick the top 5 most similar labels for the image

              similarity = (test_embeddings @ CLIP_ID_embeds.T)
              classification = (100.0 * test_embeddings @ CLIP_ID_embeds.T).softmax(dim=-1)
              unsorted_sim = {}
              unsorted_classification ={}
              for cls, sim, index in zip(classification[0], similarity[0], embeds_names):
                     unsorted_sim[index] = sim.item()
                     # unsorted_classification = classification.item()

              sorted_sim = sorted(unsorted_sim.items(), key=lambda kv: kv[1], reverse=True)
              # sorted_classification = sorted(unsorted_classification.items(), key=lambda kv: kv[1], reverse=True)
              return sorted_sim



