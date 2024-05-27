import os
import pandas as pd
from PIL import Image

import data_upload


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
    print(filepath)
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

def generated_image_checkpoint(image_path,embeds_name,alpha,guidance_scale):
    print("path entered {}".format(image_path))
    image_name = "{}*alpha:{}^GS:{}.jpg".format(embeds_name,alpha,guidance_scale)
    embeds_category = embeds_name.rsplit("_",1)[0]
    category_folder = os.path.join(image_path, embeds_category)
    os.makedirs(category_folder, exist_ok=True)
    flag, item_path = image_check(category_folder, image_name)
    return flag, item_path,image_name

def image_check(base_path, image_name):
    item_path = os.path.join(base_path, image_name)
    flag = False
    if data_upload.is_image(item_path):
        flag = True
    return flag,item_path

