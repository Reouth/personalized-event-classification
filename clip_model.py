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
              for name,img in images:
                     clip_embeddings[name] = self.image_to_embedding(img)
              return clip_embeddings

       def clip_logits_cond_img(self,path,test_image):
              clip_embeddings = self.images_to_embeddings(path)

              img_list = [image_name.replace(".jpg", "") for image_name in clip_embeddings]
              CLIP_ID_embeds = torch.cat([clip_embeddings[image_name] for image_name in clip_embeddings]).to(
                     self.device)
              CLIP_ID_embeds /= CLIP_ID_embeds.norm(dim=-1, keepdim=True)

              test_embeddings = self.image_to_embedding(test_image)
              # Pick the top 5 most similar labels for the image

              similarity = (test_embeddings @ CLIP_ID_embeds.T)
              classification = (100.0 * test_embeddings @ CLIP_ID_embeds.T).softmax(dim=-1)
              print("this is the similarity{}\n this is the classification {} this is clasffication ranked {}".format(similarity[0],classification,classification[0].topk(len(img_list))))

              unsorted_CLIP = {}
              for sim, index in zip(similarity[0], img_list):

                     unsorted_CLIP[index] = sim.item()
              sorted_CLIP = []
              sorted_CLIP = sorted(unsorted_CLIP.items(), key=lambda kv: kv[1], reverse=True)
              # sorted_CLIP[text_list[index]] = value.item()

              print(sorted_CLIP)
              return sorted_CLIP



