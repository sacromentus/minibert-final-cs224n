import pandas as pd
import argparse
import os
from datetime import datetime

def write_results_to_file(args, results):
    # Flatten the results dictionary into a list of values

    current_date = datetime.now().date()
    current_date = current_date.strftime("%Y-%m-%d")

    # Get the current time
    current_time = datetime.now().time()
    current_time = current_time.strftime("%H:%M:%S")

    data = {
        "file_name": args.filepath,
        "Pre-Loaded Model": args.load_model,
        "Pretrain/Finetune": args.option,
        "Date": current_date,
        "Time": current_time,
        "lr": args.lr,
        "Batch Size": args.batch_size,
        "Weight Decay": args.weight_decay,
        "Epochs": args.epochs,
        "Dropout Prob": args.hidden_dropout_prob
    }

    # Iterate over each category in results to flatten nested dictionaries
    data.update(results)

    # Convert the single-row data into a DataFrame
    df = pd.DataFrame([data])

    # Check if the file exists
    if os.path.exists('evaluation_results.csv'):
        # Append the DataFrame to the existing CSV file
        df.to_csv('evaluation_results.csv', mode='a', header=False, index=False)
    else:
        # Create a new CSV file and write the DataFrame to it
        df.to_csv('evaluation_results.csv', index=False)
