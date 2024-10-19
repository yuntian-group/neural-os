import pandas as pd
import os, argparse
from PIL import Image
from tqdm import tqdm
from data.data_processing.video_convert import create_padding_img, video_to_frames, sequence_creator
import multiprocessing
from functools import partial

def parse_args():
    parser = argparse.ArgumentParser(description="Converts a group of videos and their respective actions into one training dataset.")
    
    parser.add_argument("--save_dir", type=str, default='train_dataset',
                        help="directory to save the entire training set.")

    parser.add_argument("--video_dir", type=str, default='../raw_data/videos',
                        help="directory where the videos are saved.")
    
    parser.add_argument("--actions_dir", type=str, default='../raw_data/actions',
                        help="directory where the actions are saved.")
    
    parser.add_argument("--pad_start", type=bool, default=True,
                        help="Pads the start with black frames such that we can condition the first few frames with < seq_len previous frames.")

    parser.add_argument("--seq_len", type=int, default=8,
                        help="This number -1 is the number of frames conditioned on.")
    
    parser.add_argument("--bin_width", type=int, default=1,
                        help="divides the x,y cordinates by this number.")

    return parser.parse_args()

def process_video(i, args, save_dir, video_files):
    video_file = f'record_{i}.mp4'
    if video_file not in video_files:
        return None

    df = video_to_frames(
        video_path=os.path.join(args.video_dir, video_file),
        save_path=save_dir,
        actions_path=os.path.join(args.actions_dir, f'record_{i}.csv'),
        video_num=i
    )

    #Saves the sequence dataset file
    seq_df = sequence_creator(df, save_dir, seq_len=args.seq_len, bin_width=args.bin_width, pad_start=args.pad_start)
    return seq_df

if __name__ == "__main__":
    """
    Processes multiple videos sequentially and creates an entire processed dataset.
    """

    args=parse_args()

    save_dir=args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    #Create a padding image (if needed)
    black_image = create_padding_img()
    black_image.save(save_dir + '/padding.png')

    all_seqs = []

    # Determine the number of videos dynamically
    video_files = [f for f in os.listdir(args.video_dir) if f.startswith('record_') and f.endswith('.mp4')]
    n = len(video_files)

    print(f"Processing {n} videos.")

    # Create a partial function with fixed arguments
    process_video_partial = partial(process_video, args=args, save_dir=save_dir, video_files=video_files)

    # Use all available CPU cores
    num_workers = multiprocessing.cpu_count()

    # Create a multiprocessing pool
    with multiprocessing.Pool(num_workers) as pool:
        # Process videos in parallel
        results = list(tqdm(pool.imap(process_video_partial, range(n)), total=n, desc="Processing videos", unit="video"))

    # Filter out None results (skipped videos) and extend all_seqs
    all_seqs.extend([seq_df for seq_df in results if seq_df is not None])

    all_seqs_df = pd.concat(all_seqs, ignore_index=True)
    all_seqs_df.to_csv(save_dir + f'/train_dataset.csv')
