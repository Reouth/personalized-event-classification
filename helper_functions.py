import os
import pandas as pd
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
    if csv_path in csv_list:
        df_sd = pd.read_csv(csv_path)
        print(df_sd)
        if test_image in df_sd['GT image name'].unique():
            image_flag = True
    else:
        df_sd = pd.DataFrame()
    return image_flag, df_sd, csv_path