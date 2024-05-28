import os
from PIL import Image
import SD_model
import torch
import gc
import pathlib

def upload_csvs(csv_dir_path, csv_paths=[]):
  csv_dir = pathlib.Path(csv_dir_path)
  csv_paths += list(csv_dir.glob("*.csv"))
  return csv_paths

def is_image(file_path):
    """Check if a file is an image."""
    try:
        with Image.open(file_path) as img:
            return True
    except IOError:
        return False


def upload_images(base_path,class_batch=float('inf'),max_frames = float('inf')):
    """Recursively load images with their new names into a list."""
    image_data = []  # List to hold image data along with their names

    # Dictionary to store image counts for each directory
    image_counts = {}
    total_frames_count = 0
    # Iterate through items in the base directory
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)

        if os.path.isdir(item_path):
            print(f"Entering directory: {item_path}")
            # Recursively load images from subdirectories
            image_data += upload_images(item_path)
        elif is_image(item_path):
            # Get the folder name
            folder_name = os.path.basename(base_path)

            # Initialize or increment the image count for this folder
            if folder_name not in image_counts:
                image_counts[folder_name] = 0
            if total_frames_count> max_frames:
                return image_data
            if image_counts[folder_name]>class_batch:
                break
            image_counts[folder_name] += 1

            # Construct the new name for the image
            frame_number = image_counts[folder_name]
            new_name = f"{folder_name}_{frame_number:03d}.jpg"  # Zero-padded frame number

            # Append image data to the list
            image_data.append((new_name, Image.open(item_path),item_path))
        else:
            print(f"Not an image file: {item_path}")

    return image_data

# def upload_imagic_params(path,CLIP_model_name,device,loaded=[]):
#     Imagic_params = []
#     for embed_files in os.listdir(path):
#         if embed_files in loaded:
#             continue
#         else:
#             imagic_pretrained_path = os.path.join(path, embed_files)
#             print(imagic_pretrained_path)
#             if os.path.isdir(imagic_pretrained_path):
#                 print(f"uploading embeddings for directory: {imagic_pretrained_path}")
#                 pretrained_models = SD_model.SD_pretrained_load(imagic_pretrained_path, CLIP_model_name, device,
#                                                                 True)
#                 target_embeddings = torch.load(os.path.join(imagic_pretrained_path, "target_embeddings.pt")).to(device)
#                 optimized_embeddings = torch.load(os.path.join(imagic_pretrained_path, "optimized_embeddings.pt")).to(device)
#                 pipeline = SD_model.StableDiffusionPipeline(*pretrained_models)
#                 Imagic_params = (pipeline,target_embeddings,optimized_embeddings)
#                 loaded.append(embed_files)
#                 break
#     return Imagic_params, loaded

def upload_single_imagic_params(path,embeds_file,CLIP_model_name,device):
    imagic_pretrained_path = os.path.join(path, embeds_file)
    print(imagic_pretrained_path)
    if os.path.isdir(imagic_pretrained_path):
        print(f"uploading embeddings for directory: {imagic_pretrained_path}")
        pretrained_models = SD_model.SD_pretrained_load(imagic_pretrained_path, CLIP_model_name, device,
                                                        True)
        target_embeddings = torch.load(os.path.join(imagic_pretrained_path, "target_embeddings.pt")).to(device)
        optimized_embeddings = torch.load(os.path.join(imagic_pretrained_path, "optimized_embeddings.pt")).to(device)
        pipeline = SD_model.StableDiffusionPipeline(*pretrained_models)
        Imagic_params = (pipeline,target_embeddings,optimized_embeddings)
        return Imagic_params
    else:
        print('there is no embeding directory called {}'.format(imagic_pretrained_path))


def upload_embeds(path,file, CLIP_model_name,alpha, device,SD_pipe=None):
    all_embeds ={}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    imagic_parameters = upload_single_imagic_params(path, file, CLIP_model_name,
                                                                    device)
    pipeline, target_embeddings, optimized_embeddings = imagic_parameters
    if SD_pipe is not None:
        pipeline = SD_pipe

    embeddings = alpha * target_embeddings + (1 - alpha) * optimized_embeddings
    all_embeds[file] = pipeline, embeddings
    return all_embeds

def upload_cat_embeds(path, CLIP_model_name, device,SD_pipe=None):
    final_embeds = {}
    embeddings = {}

    for embeds in os.listdir(path):
        embeds_category = embeds.rsplit("_", 1)[0]

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        Imagic_params = upload_single_imagic_params(path, embeds, CLIP_model_name, device)
        _, target_embeddings, optimized_embeddings = Imagic_params
        pipeline =SD_pipe

        if embeds_category in embeddings:
            existing_target_embeds, existing_optimized_embeds, count = embeddings[embeds_category]
            embeddings[embeds_category] = (
                existing_target_embeds + target_embeddings,
                existing_optimized_embeds + optimized_embeddings,
                count + 1
            )
        else:
            embeddings[embeds_category] = (target_embeddings, optimized_embeddings, 1)

    for cat, params in embeddings.items():
        total_target_embeds, total_optimized_embeds, count = params
        embeddings = (total_target_embeds + total_optimized_embeds) / count
        final_embeds[cat] =  pipeline, embeddings

    return final_embeds








