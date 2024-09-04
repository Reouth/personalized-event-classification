import pandas as pd
import os
import glob


def extract_name(image_name):
    # Remove the "_number.jpg" part to get the base name
    return image_name.rsplit('_', 1)[0]


def calculate_average_precision(grouped_df, correct_class, loss, ascend=True):
    # Sort by loss (ascending if distance-based, descending if similarity-based)
    sorted_group = grouped_df.sort_values(by=loss, ascending=ascend)
    gallery_ids = sorted_group['gallery_id'].unique()

    # Calculate precision for each relevant item
    relevant_count = 0
    precision_list = []
    num_relevant = len(gallery_ids)  # Number of unique relevant gallery images

    for index, row in sorted_group.iterrows():
        if correct_class in row['gallery_id']:
            relevant_count += 1
            precision = relevant_count / (index + 1)
            precision_list.append(precision)

    # Average Precision (AP)
    ap = sum(precision_list) / num_relevant if num_relevant > 0 else 0
    return ap


def calculate_map_and_save_results(folder_path, output_file, clip_csv=True, avg=True):
    # Determine loss column and sorting order based on the type of CSV
    if clip_csv:
        loss = 'CLIP_loss'
        ascend = False  # Sort descending for similarity scores
        input_pred = 'input_CLIP_embeds'
    else:
        loss = 'SD_loss'
        ascend = True  # Sort ascending for distance scores
        input_pred = 'input_SD_embeds'

    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    ap_scores = []

    # Process each CSV file
    for file in csv_files:
        df = pd.read_csv(file)

        # Extract the query name from the GT image name column
        df['query_id'] = df['GT Image name'].apply(extract_name)
        df['gallery_id'] = df[input_pred].apply(extract_name)

        # Group by the query
        grouped_df = df.groupby('query_id')

        # Calculate AP for each query
        for query_id, group in grouped_df:
            correct_class = query_id.lower()  # Correct class is the query_id
            ap = calculate_average_precision(group, correct_class, loss, ascend)
            ap_scores.append(ap)

    # Mean Average Precision (mAP)
    map_value = sum(ap_scores) / len(ap_scores) if ap_scores else 0

    # Save results to CSV
    results_df = pd.DataFrame({
        'query_id': [query_id for query_id, _ in grouped_df],
        'average_precision': ap_scores
    })
    results_df.loc[len(results_df)] = ['Mean Average Precision (mAP)', map_value]  # Add mAP in the last row
    results_df.to_csv(output_file, index=False)

    return map_value



def process_value(x):
    if x.rsplit('_', 1)[-1].isdigit():  # Check if the part after the last underscore is a digit
        return x.rsplit('_', 1)[0].replace('tinaa', 'Tina')
    return x

def topk_cls_pred(df, k, correct_class, pred_column, loss, ascend, avg=True):
    df1 = df.groupby("GT Image name")
    # df1=df.groupby("Image name")

    count = 0
    correct = 0

    for _, test_image in df1:
        if avg:
            grouped = test_image.groupby(pred_column)[loss].mean()
            sorted_group = grouped.sort_values(ascending=ascend)
            top_k_pred = sorted_group.head(k).index.tolist()
        else:
            sorted_group = test_image.sort_values(by=loss, ascending=ascend)
            top_k_pred = sorted_group.head(k)[pred_column].tolist()
        print(sorted_group)
        lowercase_set = {s.lower() for s in top_k_pred}
        print("correct_class {} in lower case set {}".format(correct_class, lowercase_set))
        count += 1
        print(count)
        if any(item.startswith(correct_class.lower()) for item in lowercase_set):
            correct += 1
            print('correct{}'.format(correct))
    top_k_percentage = (correct / count) * 100
    return top_k_percentage

def csv_to_topk_results(avg,clip_csv,k_range,csvs,pred_column,results_folder):
    if avg:
        avg_name = 'avg'
    else:
        avg_name = ""
    if clip_csv:
        ascend = False
        input_pred = 'input_CLIP_embeds'
        loss = 'CLIP_loss'
        model_name = "CLIP_MODEL"
    else:
        ascend = True
        input_pred = 'input_SD_embeds'
        # input_pred = 'input_SD'
        loss = "SD_loss"
        model_name = "SD_MODEL"
    for k in k_range:
        print(k)
        top_k_all = 0
        count = 0
        results = []
        for csv in csvs:
            GT_cls = str(csv).rsplit("/", 1)[1].rsplit("_results", 1)[0]
            # GT_cls = str(csv).rsplit("/",1)[1].split("_a photo",1)[0].replace(" ","_")
            print(GT_cls)
            df = pd.read_csv(csv)
            df[pred_column] = df[input_pred].apply(lambda x: x.rsplit('_', 1)[0]).replace(" ","_")
            # df[pred_column] = df[input_pred].apply(process_value)
            # df[pred_column] = df[input_pred].apply(lambda x: x.replace(' ', '_'))
            top_k = topk_cls_pred(df, k, GT_cls, pred_column, loss, ascend, avg)
            print(top_k)
            # print("{} predicted for TOP {} accuracy : {}".format(GT_cls, k, top_k))
            results.append({"GT_cls": GT_cls, "Top_k_accuracy": top_k})
            top_k_all += top_k
            count += 1
        top_k_all = top_k_all / count
        print("MODEL TOP {} ACCURACY {}".format(k, top_k_all))
        results.append({"GT_cls": model_name, "Top_k_accuracy": top_k_all})
        results_df = pd.DataFrame(results)
        results_path = os.path.join(results_folder, "top_{}_{}_accuracy_results.csv".format(k, avg_name))
        results_df.to_csv(results_path, index=False)

def merge_csv_results(base_path,folders_names,output_dir):
    # Get the list of csv files from the first folder (assuming all folders have the same csv files)
    f1_path = os.path.join(base_path, folders_names[0])
    csv_files = [f for f in os.listdir(f1_path) if f.endswith('.csv')]

    # Dictionary to store dataframes for each csv file
    csv_data = {csv_file: [] for csv_file in csv_files}

    # Loop through each folder and each csv file
    for folder in folders_names:
        for csv_file in csv_files:
            file_path = os.path.join(base_path, folder, csv_file)
            df = pd.read_csv(file_path)
            last_row = df.iloc[-1]
            last_row['folder_name'] = folder  # Add the folder name as a new column
            csv_data[csv_file].append(last_row)

    os.makedirs(output_dir, exist_ok=True)

    for csv_file, rows in csv_data.items():
        merged_df = pd.DataFrame(rows)
        # Sort the dataframe by 'Top_k_accuracy' column in descending order
        sorted_df = merged_df.sort_values(by='Top_k_accuracy', ascending=False)
        output_path = os.path.join(output_dir, csv_file)
        sorted_df.to_csv(output_path, index=False)

    print("CSV files have been created with the last rows from each folder.")