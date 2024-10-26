import pandas as pd
import os
import ast
import shutil
from tqdm import tqdm

def convert_to_relative_coordinates(input_str):
    actions = ast.literal_eval(input_str)
    relative_actions = []
    prev_x, prev_y = None, None

    for action in actions:
        if '~' in action:
            x, y = map(float, action.split('~'))
            if prev_x is not None and prev_y is not None:
                dx = int(x) - prev_x
                dy = int(y) - prev_y
                relative_actions.append(f"{dx}~{dy}")
            else:
                relative_actions.append(f"0~0")  # No movement for the first coordinate
            prev_x, prev_y = int(x), int(y)
        else:
            relative_actions.append(action)
            prev_x, prev_y = None, None  # Reset for non-movement actions

    return str(relative_actions)

def process_csv(file_path):
    # Create a backup of the original file
    backup_path = file_path + '.backup'
    shutil.copy2(file_path, backup_path)
    print(f"Backup created at: {backup_path}")

    # Read the CSV file
    df = pd.read_csv(file_path, index_col=0)

    # Convert the coordinates in the 'actions' column to relative changes
    tqdm.pandas(desc="Converting to relative coordinates")
    df['actions'] = df['actions'].progress_apply(convert_to_relative_coordinates)

    # Save the updated DataFrame back to the original file
    df.to_csv(file_path)
    print(f"Updated file saved at: {file_path}")

if __name__ == "__main__":
    csv_path = 'train_dataset/train_dataset.csv'
    process_csv(csv_path)
