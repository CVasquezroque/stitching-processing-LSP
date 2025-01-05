import pandas as pd
import numpy as np
import h5py
import os
import argparse
import warnings

def parse_arguments():
    parser = argparse.ArgumentParser(description='Align keypoints with n-grams.')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second of the videos')
    parser.add_argument('--flatten', action='store_true', help='Keypoints are flattened (T, K*2)')
    parser.add_argument('--keypoints_dir', type=str, default='./keypoints', help='Directory containing the keypoints')
    parser.add_argument('--output_dir', type=str, default='./ngram_keypoints', help='Directory to save the n-gram keypoints')
    args = parser.parse_args()
    return args

def load_keypoints(video_name, keypoints_dir):
    """
    Load keypoints from an HDF5 file located in keypoints_dir.
    Assumes a dataset named 'keypoints' in the HDF5 file.
    Shape: (T, K, 2) or (T, K*2) depending on your data.
    """
    keypoints_path = os.path.join(keypoints_dir, f'{video_name}.hdf5')
    if not os.path.exists(keypoints_path):
        raise FileNotFoundError(f"Keypoints file not found: {keypoints_path}")
    with h5py.File(keypoints_path, 'r') as h5_file:
        keypoints_array = h5_file['keypoints'][:]
    return keypoints_array

def extract_keypoints_for_interval(keypoints_array, start_frame, end_frame):
    """
    Extract keypoints from start_frame to end_frame (inclusive).
    """
    return keypoints_array[start_frame:end_frame+1]
    
    
def build_frame_count_distribution(frames, min_frames=2, max_frames=10):
    
    counts = [0]*(max_frames+1)
    for f in frames:
        if min_frames <= f <= max_frames:
            counts[f] += 1

    total = sum(counts[min_frames:])  # sum only the valid range
    if total == 0:
        # If no data, default to uniform distribution over [min_frames..max_frames]
        valid_range_length = (max_frames - min_frames + 1)
        distribution = [0]*(min_frames) + [1/valid_range_length]*valid_range_length
    else:
        # Create a distribution that has zeros for below min_frames
        distribution = [0]*min_frames
        for f in range(min_frames, max_frames+1):
            distribution.append(counts[f]/total)
            
    return distribution
    
def sample_frame_count(distribution):
    """
    Sample a frame count from the given distribution.
    distribution: list of probabilities for frame counts [0..len(distribution)-1]
    """
    frame_values = np.arange(len(distribution))
    chosen = np.random.choice(frame_values, p=distribution)
    print(f"[DEBUG] Sampled frame count: {chosen}")
    return chosen


def create_artificial_transition(num_frames, keypoint_dim):
    """
    Create an artificial transition segment with `num_frames` frames, each keypoint value = -1.
    This simulates a transition where we do not have real frames, but we introduce artificial data.
    """
    print(f"[DEBUG] Creating artificial transition of length {num_frames}")
    return np.full((num_frames, keypoint_dim), -1, dtype=np.float32)


def generate_histograms(df):
    """
    Generate histograms for transition frames from the CSV DataFrame and
    return a probability distribution for frame counts in the range [0..10].
    """
    # Identify columns that hold transition frames
    tt_frames_cols = [col for col in df.columns if col.startswith('tt') and col.endswith('_frames')]
    all_frames = []
    for col in tt_frames_cols:
        # Collect all frame counts (drop NaN and convert to int)
        all_frames.extend(df[col].dropna().astype(int).tolist())

    # Filter to [0..10]
    filtered_frames = [frame for frame in all_frames if 0 <= frame <= 10]

    # Build and return the distribution
    distribution = build_frame_count_distribution(filtered_frames, max_frames=10)
    return distribution


def main():
    args = parse_arguments()
    FPS = args.fps
    flatten = args.flatten
    keypoint_dir = args.keypoints_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Load the n-grams csv
    ngrams_df = pd.read_csv('3grams_with_transitions.csv')
    print(ngrams_df.head())
    print(ngrams_df.columns)
    n_gram_indices = {}

    # Generate the probability distribution for transition frame counts
    frame_count_distribution = generate_histograms(ngrams_df)
    

    for idx, row in ngrams_df.iterrows():
        video_name = row['oracion'].replace('.eaf', '')
        eaf_file_name = row['oracion'].replace('.eaf', '')

        n_gram_indices.setdefault(eaf_file_name, 0)
        ngram_index = n_gram_indices[eaf_file_name]

        # Load keypoints for the video
        keypoints_array = load_keypoints(video_name, keypoint_dir )
        T_total = keypoints_array.shape[0]


        # Collect segments (glosses and transitions)
        segments, time_frames = [], []
        skip_ngram = False

        # Extract the number of intervals
        interval_columns = [col for col in row.index if col.startswith('type')]
        num_intervals = len(interval_columns)

        # lets extract the keypoints for each gloss and transition
        for i in range(1, num_intervals + 1):
            interval_type = row.get(f'type{i}')
            start_time = row.get(f'start_time{i}')
            end_time = row.get(f'end_time{i}')

            if pd.isna(interval_type) or pd.isna(start_time) or pd.isna(end_time):
                continue  # Skip if any data is missing

            start_frame = int(np.round(start_time * FPS))
            end_frame = int(np.round(end_time * FPS)) - 1  # Inclusive

            # Ensure frame indices are within bounds
            start_frame = max(0, min(start_frame, T_total - 1))
            end_frame = max(0, min(end_frame, T_total - 1))
            if end_frame < start_frame:
                end_frame = start_frame

            keypoints = extract_keypoints_for_interval(keypoints_array, start_frame, end_frame)
            segments.append((interval_type, keypoints))
            time_frames.append((start_frame, end_frame))

            if interval_type == 'transition':
                tt_frames = keypoints.shape[0]
                print(f"[DEBUG] Found a transition with {tt_frames} frames at index {idx}")
                if tt_frames >= 10:
                    print(f"[DEBUG] Skipping n-gram {idx} due to transition frames >= 10")
                    skip_ngram = True
                    break
        if skip_ngram:
            continue  # Skip processing this n-gram

        # Concatenate all segments to get the complete sequence
        complete_sequence = np.concatenate([seg[1] for seg in segments], axis=0)  # Shape: (T, K*2)
          
        # complete_sequence shape is (T, K*2) if flattened or (T, K, 2) otherwise.
  
          
        # Create the hidden sequence by setting transition keypoints to -1
        hidden_sequence = complete_sequence.copy()
        cumulative_frames = 0
        adjusted_hidden = hidden_sequence
        print(f"[DEBUG] Processing n-gram: {eaf_file_name}_{ngram_index}")
        
        for seg_type, keypoints in segments:
            frames_in_segment = keypoints.shape[0]
            start_idx = cumulative_frames
            end_idx = cumulative_frames + frames_in_segment
            
            if seg_type == 'transition':
                print(f"[DEBUG] Handling transition segment: frames_in_segment={frames_in_segment}, start_idx={start_idx}, end_idx={end_idx}")
                if frames_in_segment <= 1:
                    # Let's sample a new frame count from the distribution
                    sampled_frames = sample_frame_count(frame_count_distribution)
                    
                    keypoint_dim = adjusted_hidden.shape[1]
                    
                    # create the artificial transition segment
                    artificial_segment = create_artificial_transition(sampled_frames, keypoint_dim)
                    
                    print(f"[DEBUG] Before insertion: adjusted_hidden shape = {adjusted_hidden.shape}")
                    adjusted_hidden = np.delete(adjusted_hidden, slice(start_idx, end_idx), axis=0)
                    print(f"[DEBUG] After deletion: adjusted_hidden shape = {adjusted_hidden.shape}")
                    adjusted_hidden = np.insert(adjusted_hidden, start_idx, artificial_segment, axis=0)
                    print(f"[DEBUG] After insertion: adjusted_hidden shape = {adjusted_hidden.shape}")
                    
                    inserted_slice = adjusted_hidden[start_idx : start_idx + sampled_frames]
                    if inserted_slice.shape[0] == sampled_frames and np.all(inserted_slice == -1):
                        print("[DEBUG] Inline validation: artificial transition is correct.")
                    else:
                        print("[DEBUG] Inline validation FAILED for artificial transition.")


                    frames_in_segment = sampled_frames
                    cumulative_frames += sampled_frames
                else:
                    if 1 < frames_in_segment < 10:
                        print("[DEBUG] Replacing transition frames with -1.")
                        adjusted_hidden[start_idx:end_idx, :] = -1
                    cumulative_frames += frames_in_segment
                    
        hidden_sequence = adjusted_hidden
                
        # Create folder for this n-gram
        ngram_folder = f"{eaf_file_name}_{ngram_index}"
        ngram_path = os.path.join(output_dir, ngram_folder)
        os.makedirs(ngram_path, exist_ok=True)

        # Save the arrays
        complete_save_path = os.path.join(ngram_path, 'complete.npy')
        hidden_save_path = os.path.join(ngram_path, 'hidden.npy')
        np.save(complete_save_path, complete_sequence)
        np.save(hidden_save_path, hidden_sequence)

        print(f"Saved n-gram {ngram_index} for {eaf_file_name}")

        # Increment the n-gram index for this eaf file
        n_gram_indices[eaf_file_name] += 1

    print("Finished processing all n-grams.")

if __name__ == '__main__':
    main()