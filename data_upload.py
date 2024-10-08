import os
from PIL import Image
import SD_model
import new_SD_pipline
import torch
import gc
import pathlib
import numpy as np

def upload_csvs(csv_dir_path):
    csv_dir = pathlib.Path(csv_dir_path)
    csv_paths = list(csv_dir.glob("*.csv"))  # Create a new list
    return csv_paths

# def is_image(file_path):
#     """Check if a file is an image."""
#     try:
#         with Image.open(file_path) as img:
#             return True
#     except IOError:
#         return False

def is_image(file_path):
    """Check if a file is an image."""
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify that it is, indeed, an image
            return True
    except (IOError, SyntaxError) as e:
        print(f"File {file_path} is not an image or cannot be opened. Error: {e}")
        return False



def upload_images(base_path, class_batch=float('inf'), max_frames=float('inf')):
    """Recursively load images with their new names into a list."""
    image_data = []  # List to hold image data along with their names
    image_counts = {}  # Dictionary to store image counts for each directory
    total_frames_count = 0  # Total frame count across all directories

    # Sort the items to ensure deterministic order
    sorted_items = sorted(os.listdir(base_path))

    # If class_batch is not infinity, compute sample_tf
    if class_batch != float('inf'):
        sample_tf = np.floor(
            len([item for item in sorted_items if is_image(os.path.join(base_path, item))]) / class_batch)
        sample_tf = int(max(sample_tf, 1))  # Ensure sample_tf is at least 1
    else:
        sample_tf = 1

    # Iterate through sorted items in the base directory
    for i, item in enumerate(sorted_items):
        item_path = os.path.join(base_path, item)

        if os.path.isdir(item_path):
            print(f"Entering directory: {item_path}")
            # Recursively load images from subdirectories
            image_data += upload_images(item_path, class_batch, max_frames)
        elif is_image(item_path):
            if i % sample_tf != 0:
                continue  # Skip this file if it doesn't meet the sample_tf condition

            folder_name = os.path.basename(base_path)  # Get the folder name

            if folder_name not in image_counts:
                image_counts[folder_name] = 0

            # Check if the total frames limit or class batch limit is reached
            if total_frames_count >= max_frames or image_counts[folder_name] >= class_batch:
                print("{} frames in {} class".format(image_counts[folder_name],folder_name))
                return image_data

            image_counts[folder_name] += 1
            total_frames_count += 1

            # Construct the new name for the image
            frame_number = image_counts[folder_name]
            new_name = f"{folder_name}_{frame_number:03d}.jpg"  # Zero-padded frame number

            # Append image data to the list
            image_data.append((new_name, Image.open(item_path), item_path))
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

def upload_single_imagic_params(path,embeds_file,CLIP_model_name,device,Imagic_pipe):
    imagic_pretrained_path = os.path.join(path, embeds_file)
    print(imagic_pretrained_path)
    if os.path.isdir(imagic_pretrained_path):
        print(f"uploading embeddings for directory: {imagic_pretrained_path}")
        if Imagic_pipe:
            pretrained_models = SD_model.SD_pretrained_load(imagic_pretrained_path, CLIP_model_name, device,
                                                            True)
            pipeline = new_SD_pipline.StableDiffusionPipeline(*pretrained_models)
        else:
            pipeline =None
        target_embeddings = torch.load(os.path.join(imagic_pretrained_path, "target_embeddings.pt")).to(device)
        optimized_embeddings = torch.load(os.path.join(imagic_pretrained_path, "optimized_embeddings.pt")).to(device)

        Imagic_params = (pipeline,target_embeddings,optimized_embeddings)
        return Imagic_params
    else:
        print('there is no embeding directory called {}'.format(imagic_pretrained_path))


def upload_embeds(path,file, CLIP_model_name,alpha, device,Imagic_pipe,SD_pretrained=None):
    all_embeds ={}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    imagic_parameters = upload_single_imagic_params(path, file, CLIP_model_name,
                                                                    device,Imagic_pipe)

    if SD_pretrained is not None:
        pipeline = new_SD_pipline.StableDiffusionPipeline(*SD_pretrained)
        _,target_embeddings, optimized_embeddings = imagic_parameters
    else:
        pipeline, target_embeddings, optimized_embeddings = imagic_parameters
    embeddings = alpha * target_embeddings + (1 - alpha) * optimized_embeddings
    all_embeds[file] = pipeline, embeddings
    return all_embeds

def upload_cat_embeds(path, CLIP_model_name, device,Imagic_pipe,SD_pipe):
    final_embeds = {}
    embeddings = {}

    for embeds in os.listdir(path):
        embeds_category = embeds.rsplit("_", 1)[0]

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        Imagic_params = upload_single_imagic_params(path, embeds, CLIP_model_name, device,Imagic_pipe)
        _, target_embeddings, optimized_embeddings = Imagic_params
        pipeline = new_SD_pipline.StableDiffusionPipeline(*SD_pipe)

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








