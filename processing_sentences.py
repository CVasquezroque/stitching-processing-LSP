import os
import argparse
import warnings
import numpy as np
import pandas as pd
import h5py
import pympi
import random
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Process entire videos (no new frames), discarding transitions > maxTransitionFrames.
        For transitions < minTransitionFrames, re-label to have at least that many frames (Option A)."""
    )
    parser.add_argument('--rootPath', type=str, default='./',
                        help='Root path to look for EAF files.')
    parser.add_argument('--fileNameContains', type=str, default='ORACION',
                        help='Substring that EAF filenames must contain (e.g. ORACION).')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Frames per second of the videos.')
    parser.add_argument('--maxTransitionFrames', type=int, default=10,
                        help='Discard the entire video if any transition exceeds this number of frames.')
    parser.add_argument('--minTransitionFrames', type=int, default=2,
                        help='If a transition is shorter than this, re-label it to have >= minTransitionFrames..maxTransitionFrames frames.')
    parser.add_argument('--keypoints_dir', type=str, default='./keypoints',
                        help='Directory with .hdf5 keypoints (one per video).')
    parser.add_argument('--output_dir', type=str, default='./resampled_videos_output',
                        help='Folder where we store final (complete.npy, hidden.npy).')
    parser.add_argument('--plot_filename', type=str, default='transition_distribution.png',
                        help='File name for saving the distribution plot.')
    parser.add_argument('--json_output', type=str, default='relabeling_info.json',
                        help='File name to store the final transitions info (JSON).')
    args = parser.parse_args()
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def find_eaf_files(root_path, file_name_contains):
    eaf_files = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        for fname in filenames:
            if fname.endswith('.eaf') and file_name_contains in fname:
                eaf_files.append(os.path.join(dirpath, fname))
    return eaf_files

def load_keypoints(video_name, keypoints_dir):
    path = os.path.join(keypoints_dir, f"{video_name}.hdf5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Keypoints file not found: {path}")
    with h5py.File(path, 'r') as f:
        arr = f['keypoints'][:]
    return arr

def process_eaf_file(eaf_path):
    eaf_obj = pympi.Elan.Eaf(eaf_path)
    eaf_file_name = os.path.basename(eaf_path)
    main_tier = 'GLOSA_IA'
    if main_tier not in eaf_obj.get_tier_names():
        return [], eaf_file_name
    raw_anns = eaf_obj.get_annotation_data_for_tier(main_tier)
    annotations = []
    for ann in raw_anns:
        start_s = ann[0] / 1000.0
        end_s   = ann[1] / 1000.0
        gloss   = ann[2]
        annotations.append({
            'gloss': gloss,
            'start_time': start_s,
            'end_time': end_s
        })
    annotations.sort(key=lambda x: x['start_time'])
    return annotations, eaf_file_name

def gather_transition_frames(eaf_files, fps, min_f, max_f):
    """Collect transitions in [min_f..max_f] to build a distribution."""
    frames_list = []
    for ef in eaf_files:
        ann, _ = process_eaf_file(ef)
        for i in range(len(ann)-1):
            dur = ann[i+1]['start_time'] - ann[i]['end_time']
            if dur < 0: 
                continue
            f = int(round(dur * fps))
            if min_f <= f <= max_f:
                frames_list.append(f)
    return frames_list

def build_distribution(frames_list, min_f, max_f):
    counts = [0]*(max_f+1)
    for v in frames_list:
        counts[v] += 1
    total = sum(counts[min_f:max_f+1])
    dist = [0]*(max_f+1)
    if total == 0:
        length = (max_f - min_f + 1)
        for i in range(min_f, max_f+1):
            dist[i] = 1.0/length
    else:
        for i in range(min_f, max_f+1):
            dist[i] = counts[i]/total
    return dist

def sample_transition_length(dist, min_f, max_f):
    frame_vals = np.arange(len(dist))
    chosen = np.random.choice(frame_vals, p=dist)
    while not (min_f <= chosen <= max_f):
        chosen = np.random.choice(frame_vals, p=dist)
    return chosen

def plot_dist(dist, min_f, max_f, filename):
    import matplotlib.pyplot as plt
    x_vals = np.arange(min_f, max_f+1)
    y_vals = [dist[x] for x in x_vals]
    plt.figure(figsize=(5,3))
    plt.bar(x_vals, y_vals, color='orange', edgecolor='black')
    plt.title("Transition Distribution")
    plt.xlabel("Frames")
    plt.ylabel("Probability")
    for i, v in enumerate(y_vals):
        plt.text(x_vals[i], v+0.002, f"{v:.2f}", ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def print_transition_info(annotations, fps, max_f):
    """Return False if any transition > max_f, otherwise True."""
    valid = True
    for i in range(len(annotations)-1):
        dur = annotations[i+1]['start_time'] - annotations[i]['end_time']
        f = int(round(dur*fps))
        print(f"   [INFO] Transition from {i} to {i+1}: {f} frames")
        if f > max_f:
            valid = False
    return valid

def resegment_gloss_opcionA(ann, i, new_t_len, fps):
    """Force next gloss to start after new_t_len frames from gloss i end."""
    glossA_end   = ann[i]['end_time']
    glossB_start = ann[i+1]['start_time']
    dur_old = glossB_start - glossA_end
    if dur_old < 0:
        return
    oldA_endF = int(round(glossA_end*fps))
    oldB_stF  = int(round(glossB_start*fps))
    # Opcion A: glosa i termina un poco antes, glosa i+1 empieza oldA_endF + new_t_len
    # Podriamos "robar" frames mitad y mitad, pero aqui forzamos un hueco total de new_t_len
    # sin solapamiento.
    m = new_t_len // 2
    # Optionally you could do that, or do a different ratio. We'll do a simple approach:
    A_endF = oldA_endF - m
    if A_endF < 0:
        A_endF = 0
    B_startF = A_endF + new_t_len
    if B_startF <= A_endF:
        return
    newA_end_s   = A_endF/fps
    newB_start_s = B_startF/fps
    print(f"   [DEBUG][OpcionA] oldTransition={dur_old:.3f}s => new={new_t_len} frames")
    print(f"   [DEBUG][OpcionA] A_endF => {oldA_endF} -> {A_endF}, B_startF => {oldB_stF} -> {B_startF}")
    ann[i]['end_time']     = newA_end_s
    ann[i+1]['start_time'] = newB_start_s

def main():
    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1) gather EAF
    eaf_files = find_eaf_files(args.rootPath, args.fileNameContains)
    print(f"[INFO] Found {len(eaf_files)} EAF files.")

    # 2) build distribution in [minTransitionFrames..maxTransitionFrames]
    frames_list = gather_transition_frames(eaf_files,
                                           args.fps,
                                           args.minTransitionFrames,
                                           args.maxTransitionFrames)
    dist = build_distribution(frames_list,
                              args.minTransitionFrames,
                              args.maxTransitionFrames)
    plot_dist(dist, args.minTransitionFrames,
              args.maxTransitionFrames,
              args.plot_filename)
    print(f"[INFO] Built distribution in [{args.minTransitionFrames}..{args.maxTransitionFrames}]")
    print(f"[INFO] Plot saved to {args.plot_filename}\n")

    total_ok = 0
    total_discarded = 0
    total_short_fixed = 0

    relabel_json = {}

    for eaf_path in eaf_files:
        ann, eaf_file = process_eaf_file(eaf_path)
        print(f"\n[INFO] Processing EAF: {eaf_file}")
        if not ann:
            print("   [WARN] No valid annotations. Discarding.")
            total_discarded += 1
            continue

        original_trans_count = max(0, len(ann)-1)
        print(f"[DEBUG] Checking transitions in {eaf_file}:")
        is_valid = print_transition_info(ann, args.fps, args.maxTransitionFrames)
        if not is_valid:
            print(f"   [WARN] There's a transition > {args.maxTransitionFrames} frames => discard {eaf_file}")
            total_discarded += 1
            continue

        short_fixed = 0
        for i in range(len(ann)-1):
            dur_s = ann[i+1]['start_time'] - ann[i]['end_time']
            frames = int(round(dur_s * args.fps))
            if frames < 0: 
                continue
            if frames < args.minTransitionFrames:
                new_len = sample_transition_length(dist,
                                                   args.minTransitionFrames,
                                                   args.maxTransitionFrames)
                print(f"   [INFO] Re-label short transition from gloss {i} to {i+1}: {frames} => {new_len}")
                resegment_gloss_opcionA(ann, i, new_len, args.fps)
                short_fixed += 1
        total_short_fixed += short_fixed

        # 3) Load keypoints
        video_name = eaf_file.replace('.eaf','')
        try:
            kp_arr = load_keypoints(video_name, args.keypoints_dir)
        except FileNotFoundError:
            print(f"   [WARN] No keypoints for {video_name}, discarding.")
            total_discarded += 1
            continue

        T_total = kp_arr.shape[0]
        print(f"[DEBUG] {video_name}: total frames={T_total}")

        # 4) final build of complete + hidden
        seg_complete = []
        seg_hidden   = []
        transitions_info = []

        for i, a in enumerate(ann):
            gloss_start_f = int(round(a['start_time'] * args.fps))
            gloss_end_f   = int(round(a['end_time']   * args.fps))

            gloss_start_f = max(0, min(gloss_start_f, T_total-1))
            gloss_end_f   = max(0, min(gloss_end_f,   T_total-1))

            print(f"[DEBUG] Gloss {i} => frames={gloss_start_f}..{gloss_end_f}")
            block_gloss = kp_arr[gloss_start_f : gloss_end_f+1]
            seg_complete.append(block_gloss)
            seg_hidden.append(block_gloss)

            if i < len(ann)-1:
                next_start_f = int(round(ann[i+1]['start_time'] * args.fps))
                next_start_f = max(0, min(next_start_f, T_total-1))

                trans_start_f = gloss_end_f + 1
                if trans_start_f <= next_start_f:
                    # We'll define the transition as [trans_start_f.. next_start_f],
                    # so we do block_trans = kp_arr[trans_start_f : next_start_f+1]
                    trans_block = kp_arr[trans_start_f : next_start_f+1]

                    print(f"[DEBUG] Transition {i} => frames={trans_start_f}..{next_start_f}")
                    seg_complete.append(trans_block)
                    block_hidden = np.full(trans_block.shape, -1, dtype=np.float32)
                    seg_hidden.append(block_hidden)

                    real_len = (next_start_f - trans_start_f + 1)
                    transitions_info.append({
                        "index": i,
                        "start_frame": trans_start_f,
                        "end_frame": next_start_f,
                        "length_frames": real_len
                    })

        if not seg_complete:
            print("   [WARN] No segments built => discarding.")
            total_discarded += 1
            continue

        final_complete = np.concatenate(seg_complete, axis=0)
        final_hidden   = np.concatenate(seg_hidden,   axis=0)
        if final_complete.shape != final_hidden.shape:
            print(f"   [WARN] shape mismatch => discarding.")
            total_discarded += 1
            continue

        # 5) Save outputs
        out_folder = os.path.join(args.output_dir, video_name)
        os.makedirs(out_folder, exist_ok=True)
        np.save(os.path.join(out_folder, 'complete.npy'), final_complete)
        np.save(os.path.join(out_folder, 'hidden.npy'),   final_hidden)

        final_trans_count = max(0, len(ann)-1)
        print(f"[INFO] {video_name}: original={original_trans_count}, final={final_trans_count}, short_fixed={short_fixed}")
        print(f"[INFO] Saved => {out_folder}, shape={final_complete.shape}")

        relabel_json[video_name] = {
            "original_transitions": original_trans_count,
            "final_transitions": final_trans_count,
            "short_fixed": short_fixed,
            "transitions": transitions_info
        }

        total_ok += 1

    # Summary
    print("\n===== FINAL SUMMARY =====")
    print(f"Videos OK:        {total_ok}")
    print(f"Videos Discarded: {len(eaf_files)-total_ok}")
    print(f"Short transitions fixed: {total_short_fixed}")
    print("=========================\n")

    # 6) Save JSON
    json_path = os.path.join(args.output_dir, args.json_output)
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(relabel_json, jf, indent=2)
    print(f"[INFO] JSON saved to {json_path}")

if __name__ == '__main__':
    set_seed(42)
    main()
