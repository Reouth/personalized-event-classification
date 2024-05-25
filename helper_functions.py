import os
import pandas as pd
from PIL import Image
def list_csv_files_in_directory(directory_path):
    try:
        # List all items in the directory
        items = os.listdir(directory_path)

        # Filter out only the CSV files
        csv_files = [item for item in items if item.endswith('.csv') and os.path.isfile(os.path.join(directory_path, item))]

        return csv_files
    except FileNotFoundError:
        print(f"The directory {directory_path} does not exist.")
        return []

def csv_checkpoint(csv_folder,cls,test_image):
    image_flag =False
    csv_list = list_csv_files_in_directory(csv_folder)
    filepath = "{}_results.csv".format(cls)
    csv_path = os.path.join(csv_folder, filepath)
    print(csv_list)
    print(csv_path)
    if filepath in csv_list:
        df_sd = pd.read_csv(csv_path)
        df_sd = df_sd.drop(columns=['Unnamed: 0'])
        df_sd.columns = [col.strip() for col in df_sd.columns]


        # Check if the column name 'GT Image name' exists in the DataFrame columns
        if 'GT Image name' in df_sd.columns:
            if test_image in df_sd['GT Image name'].unique():
                print(f"test_image: {test_image} already in csv.")
                image_flag = True
        else:
            print("'GT Image name' column is not found in the CSV file.")
            print("Available columns:", df_sd.columns)
        # print(df_sd.columns)
        # if 'GT image name' in list(df_sd.columns):
        #     if test_image in df_sd['GT image name'].unique():
        #         "test_image: {} already in csv.".format(test_image)
        #         image_flag = True
    else:
        df_sd = pd.DataFrame()
    return image_flag, df_sd, csv_path

def generated_image_checkpoint(embeds_path,image_path,alpha,guidance_scale):
    for embed_files in os.listdir(embeds_path):
        image_name = "{}*alpha:{}^GS:{}.jpg".format(embed_files,alpha,guidance_scale)
        embeds_category = embed_files.rsplit("_",1)[0]
        category_folder = os.path.join(image_path, embeds_category)
        os.makedirs(category_folder, exist_ok=True)
        return image_check(category_folder, image_name),embed_files

def image_check(base_path, image_name):
    item_path=""
    flag = False

    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)

        if os.path.isdir(item_path):
            print(f"Entering directory: {item_path}")
            # Recursively load images from subdirectories
            flag = image_check(item_path, image_name) or flag
        elif item == image_name:
            flag = True
            print("image {} already generated".format(image_name))
    return flag,item_path

