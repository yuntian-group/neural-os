import pandas as pd

# Load your CSV file
df = pd.read_csv('train_dataset/train_dataset.target_frames.csv')

# Find max image_num for each record_num
max_image_num = df.groupby('record_num')['image_num'].max().reset_index()

# Find 5 record_nums with smallest max image_num
smallest_records = max_image_num.nsmallest(5, 'image_num')

# Print results
print(smallest_records)
