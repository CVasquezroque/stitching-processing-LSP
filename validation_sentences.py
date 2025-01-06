import os
import json
import numpy as np

def validate_pipeline(json_path, output_dir):
    # Final summary counters
    total_videos = 0
    validated_videos = 0
    total_errors = 0

    # Load the JSON that was created after re-labeling
    if not os.path.exists(json_path):
        print(f"[ERROR] JSON does not exist: {json_path}")
        return
    with open(json_path, 'r', encoding='utf-8') as jf:
        relabel_info = json.load(jf)

    total_videos = len(relabel_info)

    for video_name, info in relabel_info.items():
        video_dir = os.path.join(output_dir, video_name)
        complete_path = os.path.join(video_dir, 'complete.npy')
        hidden_path   = os.path.join(video_dir, 'hidden.npy')

        # Check existence
        if not os.path.exists(complete_path) or not os.path.exists(hidden_path):
            print(f"[ERROR] Missing .npy files for {video_name}")
            if not os.path.exists(complete_path):
                print(f"   - {complete_path} is missing")
            if not os.path.exists(hidden_path):
                print(f"   - {hidden_path} is missing")
            total_errors += 1
            continue

        # Load arrays
        try:
            complete_arr = np.load(complete_path)
            hidden_arr   = np.load(hidden_path)
        except Exception as e:
            print(f"[ERROR] Could not load arrays for {video_name}: {e}")
            total_errors += 1
            continue

        # Shape check
        if complete_arr.shape != hidden_arr.shape:
            print(f"[ERROR] shape mismatch in {video_name}")
            print(f"   - complete.shape={complete_arr.shape}, hidden.shape={hidden_arr.shape}")
            total_errors += 1
            continue

        transitions = info.get('transitions', [])
        for t in transitions:
            start_f = t['start_frame']
            end_f   = t['end_frame']

            # Range check
            if not (0 <= start_f <= end_f < hidden_arr.shape[0]):
                print(f"[WARN] Out-of-bounds transition for {video_name}: {start_f}..{end_f}")
                total_errors += 1
                continue

            region_hidden   = hidden_arr[start_f:end_f+1]
            region_complete = complete_arr[start_f:end_f+1]

            # Must be -1 in hidden
            if not np.allclose(region_hidden, -1, atol=1e-6):
                print(f"[ERROR] Hidden region not fully masked for {video_name}")
                print(f"   - frames={start_f}..{end_f} contain non--1 values")
                total_errors += 1

            # Must not be all -1 in complete
            if np.allclose(region_complete, -1, atol=1e-6):
                print(f"[ERROR] Complete region is also all -1 in {video_name}")
                print(f"   - frames={start_f}..{end_f} contain only -1 values")
                total_errors += 1

        validated_videos += 1

    # Final summary
    print("\n===== VALIDATION SUMMARY =====")
    print(f"Videos in JSON:      {total_videos}")
    print(f"Videos validated:    {validated_videos}")
    print(f"Total errors found:  {total_errors}")
    print("================================\n")

if __name__ == "__main__":
    json_path = "./full_output_corrected_final/relabeling_info.json"  # Or your path
    output_dir = "./full_output_corrected_final"                      # Or your path
    validate_pipeline(json_path, output_dir)
