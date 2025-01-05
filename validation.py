import os
import numpy as np
import pandas as pd

# Configuration
NGRAM_OUTPUT_DIR = './ngram_keypoints'
CSV_FILE = '3grams_with_transitions.csv'
FPS = 30  # Match the FPS used during processing

def main():
    # Load the n-grams DataFrame
    if not os.path.isfile(CSV_FILE):
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)

    # Build an index to easily retrieve rows by (video_name, ngram_index)
    # We know from the main script: ngram folders are named like <video_name>_<ngram_index>
    # We'll try to deduce (video_name, ngram_index) from directory names.
    # Note: ngram_index was assigned in order of appearance for each video_name in the main script.
    # Without the exact indexing logic from main script, we rely on directory structure.
    
    ngram_dirs = [d for d in os.listdir(NGRAM_OUTPUT_DIR) if os.path.isdir(os.path.join(NGRAM_OUTPUT_DIR, d))]
    if not ngram_dirs:
        print("No n-gram directories found in output directory. Nothing to validate.")
        return
    
    # Group the DataFrame by 'oracion' to handle indexing logic consistently
    grouped = df.groupby(df['oracion'])

    for ngram_dir in ngram_dirs:
        ngram_path = os.path.join(NGRAM_OUTPUT_DIR, ngram_dir)
        
        # Parse video_name and ngram_index from directory name
        # The naming was: <video_name>_<ngram_index>
        parts = ngram_dir.split('_')
        if len(parts) < 2:
            print(f"Skipping {ngram_dir}: cannot parse video_name and ngram_index.")
            continue
        video_name = "_".join(parts[:-1])  # Handle if video_name has underscores
        try:
            ngram_index = int(parts[-1])
        except ValueError:
            print(f"Skipping {ngram_dir}: last part of name is not an integer index.")
            continue
        
        oracion_name = video_name + '.eaf'

        if oracion_name not in grouped.groups:
            print(f"No entries in CSV for {oracion_name}.")
            continue
        
        # Retrieve the rows for this video
        video_rows = grouped.get_group(oracion_name)
        # Sort by a stable index (assumes the order of n-grams was sequential)
        video_rows = video_rows.reset_index(drop=True)
        
        if ngram_index >= len(video_rows):
            print(f"N-gram index {ngram_index} out of range for {oracion_name}.")
            continue
        
        row = video_rows.iloc[ngram_index]
        
        complete_file = os.path.join(ngram_path, 'complete.npy')
        hidden_file = os.path.join(ngram_path, 'hidden.npy')

        if not os.path.exists(complete_file) or not os.path.exists(hidden_file):
            print(f"Missing files in {ngram_dir}. Skipping...")
            continue

        complete_seq = np.load(complete_file)
        hidden_seq = np.load(hidden_file)

        # Now let's re-derive the segments and check transitions
        interval_columns = [c for c in row.index if c.startswith('type')]
        num_intervals = len(interval_columns)

        segments = []
        T_total = complete_seq.shape[0]  # This should match sum of all segments from CSV intervals

        for i in range(1, num_intervals + 1):
            interval_type = row.get(f'type{i}')
            start_time = row.get(f'start_time{i}')
            end_time = row.get(f'end_time{i}')

            if pd.isna(interval_type) or pd.isna(start_time) or pd.isna(end_time):
                continue

            start_frame = int(round(start_time * FPS))
            end_frame = int(round(end_time * FPS)) - 1
            # Clip these to T_total if needed (not strictly necessary)
            start_frame = max(0, min(start_frame, T_total - 1))
            end_frame = max(0, min(end_frame, T_total - 1))
            if end_frame < start_frame:
                end_frame = start_frame

            frames_in_segment = end_frame - start_frame + 1
            segments.append((interval_type, start_frame, end_frame, frames_in_segment))

        # Validate each transition in hidden_seq according to logic:
        # 1. If frames_in_segment <= 0: should have inserted an artificial segment of random length (0 to 10) of all -1.
        # 2. If 0 < frames_in_segment <= 10: should have replaced them with -1 in hidden_seq.
        # 3. If frames_in_segment > 10, the main script should have skipped the n-gram altogether.

        # We'll track cumulative frames in hidden_seq and verify transitions.
        cumulative_frames = 0
        validation_passed = True

        for seg_type, start_frame, end_frame, frames_in_segment in segments:
            segment_length = frames_in_segment
            seg_hidden = hidden_seq[cumulative_frames:cumulative_frames+segment_length]

            if seg_type == 'transition':
                if frames_in_segment <= 1:
                    # Artificial segment should have been inserted.
                    # We do not know the exact sampled length, but we know it should be > 0 and <= 10
                    # and all -1.
                    # Let's just check that it is all -1 and length is between 1 and 10.
                    if (len(seg_hidden) < 1 or len(seg_hidden) > 10 or not np.all(seg_hidden == -1)):
                        print(f"[FAIL] {ngram_dir}: Transition with frames_in_segment={frames_in_segment} did not produce a valid artificial segment.")
                        print(len(seg_hidden))
                        print(seg_hidden)
                        validation_passed = False
                    else:
                        print(f"[OK] {ngram_dir}: Artificial transition inserted correctly ({len(seg_hidden)} frames all -1).")

                elif 1 < frames_in_segment <= 10:
                    # Should be replaced with -1 frames
                    if (seg_hidden.shape[0] == frames_in_segment) and np.all(seg_hidden == -1):
                        print(f"[OK] {ngram_dir}: Transition with {frames_in_segment} frames replaced with -1 as expected.")
                    else:
                        print(f"[FAIL] {ngram_dir}: Transition with {frames_in_segment} frames not properly replaced with -1.")
                        validation_passed = False
                else:
                    # frames_in_segment > 10 n-grams should have been skipped by the main code,
                    # so we shouldn't even be here.
                    # Just a sanity check.
                    if frames_in_segment > 10:
                        print(f"[WARN] {ngram_dir}: Found a transition with >10 frames. This should have been skipped.")
                        # Not necessarily fail here since main code should have skipped it.
                        validation_passed = False

            # Move to next segment
            cumulative_frames += seg_hidden.shape[0]

        # Final shape check
        if hidden_seq.shape[0] < complete_seq.shape[0]:
            print(f"[FAIL] {ngram_dir}: Hidden sequence shorter than complete sequence. Something is off.")
            validation_passed = False

        if validation_passed:
            print(f"[PASS] {ngram_dir}: Validation passed.")
        else:
            print(f"[FAIL] {ngram_dir}: Validation failed for one or more conditions.")

if __name__ == '__main__':
    main()
