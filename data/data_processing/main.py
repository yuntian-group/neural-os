import pandas as pd
import os, argparse
from PIL import Image
from tqdm import tqdm
from data.data_processing.video_convert import create_padding_img, video_to_frames, sequence_creator
import multiprocessing
from functools import partial
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="Converts a group of videos and their respective actions into one training dataset.")
    
    parser.add_argument("--save_dir", type=str, default='train_dataset',
                        help="directory to save the entire training set.")

    parser.add_argument("--video_dir", type=str, default='../data_collection/raw_data/raw_data/videos',
                        help="directory where the videos are saved.")
    
    parser.add_argument("--actions_dir", type=str, default='../data_collection/raw_data/raw_data/actions',
                        help="directory where the actions are saved.")
    
    parser.add_argument("--seq_len", type=int, default=1,
                        help="This number -1 is the number of frames conditioned on.")

    args = parser.parse_args()
    
    # Add directory validation
    for dir_path in [args.video_dir, args.actions_dir]:
        if not os.path.exists(dir_path):
            raise ValueError(f"Directory does not exist: {dir_path}")
            
    return args

def process_video(i: int, args: argparse.Namespace, save_dir: str, video_files: list[str]) -> pd.DataFrame | None:
    """
    Process a single video file and create sequences.
    
    Args:
        i: Video index
        args: Command line arguments
        save_dir: Directory to save processed data
        video_files: List of available video files
        
    Returns:
        DataFrame containing sequences or None if video not found
    """
    video_file = f'record_{i}.mp4'
    if video_file not in video_files:
        return None

    try:
        df = video_to_frames(
            video_path=os.path.join(args.video_dir, video_file),
            save_path=save_dir,
            actions_path=os.path.join(args.actions_dir, f'record_{i}.csv'),
            video_num=i
        )
        
        seq_df = sequence_creator(df, save_dir, seq_len=args.seq_len)
        del df  # Clean up memory
        return seq_df
        
    except Exception as e:
        print(f"Error processing video {video_file}: {str(e)}")
        raise e

def get_video_dimensions(video_path: str) -> tuple[int, int]:
    """
    Safely get video dimensions using OpenCV.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (width, height)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        ret, frame = cap.read()
        cap.release()  # Properly release the video capture
        
        if not ret:
            raise ValueError(f"Could not read frame from video file: {video_path}")
        
        height, width = frame.shape[:2]
        return width, height
        
    except Exception as e:
        raise ValueError(f"Error reading video dimensions: {str(e)}")

if __name__ == "__main__":
    """
    Processes multiple videos sequentially and creates an entire processed dataset.
    """

    args = parse_args()
    
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Improve video file pattern matching
    video_files = [f for f in os.listdir(args.video_dir) 
                  if f.startswith('record_') and f.endswith('.mp4')
                  and os.path.isfile(os.path.join(args.video_dir, f))]
    
    if not video_files:
        raise ValueError(f"No valid video files found in {args.video_dir}")
    
    # Get video dimensions using the new helper function
    first_video = os.path.join(args.video_dir, video_files[0])
    width, height = get_video_dimensions(first_video)
    print(f"Video dimensions: width: {width}, height: {height}")

    # Create a padding image with the same size
    black_image = create_padding_img(width, height)
    black_image.save(os.path.join(save_dir, 'padding.png'))

    all_seqs = []

    n = len(video_files)
    print(f"Processing {n} videos.")

    # Create a partial function with fixed arguments
    process_video_partial = partial(process_video, args=args, save_dir=save_dir, video_files=video_files)

    # Use all available CPU cores (removed hardcoding to 1)
    num_workers = min(multiprocessing.cpu_count(), 64)
    print(f"Using {num_workers} workers.")

    process_video_partial(0)
    sys.exit(1)

    try:
        # Create a multiprocessing pool
        with multiprocessing.Pool(num_workers) as pool:
            # Process videos in parallel
            results = list(tqdm(
                pool.imap(process_video_partial, range(n)), 
                total=n, 
                desc="Processing videos", 
                unit="video"
            ))

        # Filter out None results and combine sequences
        all_seqs = [seq_df for seq_df in results if seq_df is not None]
        
        if not all_seqs:
            raise ValueError("No sequences were successfully processed")
            
        all_seqs_df = pd.concat(all_seqs, ignore_index=True)
        all_seqs_df.to_csv(os.path.join(save_dir, 'train_dataset.csv'))
        
        # Clean up memory
        del results
        del all_seqs
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
