import pandas as pd
import os


def calculate_average_precision(input_folder, output_folder, clip_csv=False):
    results = {}

    # Iterate through all CSV files in the input folder
    for csv_file in os.listdir(input_folder):
        if csv_file.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_folder, csv_file))
            gallery_class = csv_file.split('_')[0]  # Extract the gallery class from the filename
            print(f'Processing {gallery_class} class...')

            # Group by GT Image name (the gallery images)
            grouped_df = df.groupby("GT Image name")

            for gt_image, group in grouped_df:
                # Get the top query (the one with the best score)
                if clip_csv:
                    top_query = group.loc[group['CLIP_loss'].idxmax()]  # Use idxmax for CLIP
                else:
                    top_query = group.loc[group['SD_loss'].idxmin()]  # Use idxmin for SD

                top_query_id = top_query['input_CLIP_embeds'].rsplit('_', 1)[0] if clip_csv else \
                top_query['input_SD_embeds'].rsplit('_', 1)[0]  # Extract query name without number

                # Initialize counts for relevant and total matches
                if top_query_id not in results:
                    results[top_query_id] = {'relevant': 0, 'total': 0}

                # Check if the top query matches the gallery class
                if gt_image.startswith(top_query_id):  # Match in the same class
                    results[top_query_id]['relevant'] += 1

                # Count total matches for the query across all classes
                results[top_query_id]['total'] += 1

    # Calculate Average Precision for each query
    ap_scores = {}
    for query_id, counts in results.items():
        ap = counts['relevant'] / counts['total'] if counts['total'] > 0 else 0
        ap_scores[query_id] = ap

    # Calculate Mean Average Precision (mAP)
    map_score = sum(ap_scores.values()) / len(ap_scores) if ap_scores else 0

    # Create a DataFrame for results and save to CSV
    results_df = pd.DataFrame(list(ap_scores.items()), columns=['query_id', 'average_precision'])
    results_df.loc[len(results_df.index)] = ['mAP', map_score]

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save results to the output folder
    results_df.to_csv(os.path.join(output_folder, 'average_precision_results.csv'), index=False)

    return ap_scores, map_score


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