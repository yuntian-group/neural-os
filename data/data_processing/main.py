import pandas as pd
import os
from PIL import Image
from data.data_processing.video_convert import video_to_frames, sequence_creator

if __name__ == "__main__":

    """
    Processes multiple videos sequentially and creates an entire processed dataset.
    """

    save_path='train_dataset_1000_videos'
    os.makedirs(save_path, exist_ok=True)

    #Create a padding image (if needed)
    black_image = Image.new('RGB', (256, 256), color=(0, 0, 0))
    black_image.save(save_path + '/padding.png')

    all_seqs = []

    n = 1000

    print(f"Proccessing {n} videos.")

    for i in range(n):
        if i% 50 == 49: print(f"\u2705 Proccessed 50 videos... {n-i-1} remaining.")

        df = video_to_frames(
            video_path=f'../raw_data/videos/record_{i}.mp4',
            save_path=save_path,
            actions_path=f'../raw_data/actions/record_{i}.csv',
            video_num=i
        )

        #Saves the sequence dataset file
        seq_df = sequence_creator(df, save_path, seq_len=8, bin_width=4, pad_start=True)
        all_seqs.append(seq_df)

    all_seqs_df = pd.concat(all_seqs, ignore_index=True)
    all_seqs_df.to_csv(save_path + f'/train_dataset.csv')