import pandas as pd


def calculate_top_k_accuracy(csv_path, correct_class_col, pred_class_col, loss_col, k,down):
    df = pd.read_csv(csv_path)
    cls = csv_path.rsplit("/",1)[1].rsplit("_results",1)[0]
    correct, counts = 0, 0
    group_accuracies = {}

    grouped = df.groupby(correct_class_col)

    # Iterate over each group
    for image_id, group in grouped:

        group = group.sort_values(by=loss_col, ascending=down)
        top_k_predictions = group[pred_class_col].head(k).tolist()
        correct_class = group[correct_class_col].iloc[0]
        if correct_class in top_k_predictions:
            group_correct = 1
            correct += 1
        else:
            group_correct = 0

        counts += 1

        # Store the accuracy for the current group
        group_accuracies[image_id] = group_correct * 100

    # Calculate the total top-k accuracy
    total_top_k_accuracy = (correct / counts) * 100

    return group_accuracies, total_top_k_accuracy