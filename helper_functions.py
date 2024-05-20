import os
import pandas as pd
def list_folders_in_directory(directory_path):
    try:
        # List all items in the directory
        items = os.listdir(directory_path)

        # Filter out only the folders
        folders = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]

        return folders
    except FileNotFoundError:
        print(f"The directory {directory_path} does not exist.")
        return []

def csv_checkpoint(csv_folder,cls,test_image):
    image_flag =False
    cls_list = list_folders_in_directory(csv_folder)
    filepath = "{}_results.csv".format(cls)
    csv_path = os.path.join(csv_folder, filepath)
    if csv_path in cls_list:
        df_sd = pd.read_csv(csv_path)
        if test_image in df_sd['GT image name'].unique():
            image_flag = True
    else:
        df_sd = pd.DataFrame()
    return image_flag, df_sd, csv_path