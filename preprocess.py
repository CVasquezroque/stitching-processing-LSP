import os
import warnings
import pandas as pd
import numpy as np
import cv2
import time
import h5py
import argparse

from utils.pucp_glosas_video_reader import get_pucp_glosas_data
from utils import mediapipe_functions

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process videos to extract keypoints.')
    parser.add_argument('--flatten', action='store_true', help='Flatten keypoints to (T, K*2)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second of the videos')
    parser.add_argument('--video_dir', type=str, default='./datasets/pucp_videos', help='Directory containing the videos')
    args = parser.parse_args()
    return args


def get_keypoints(model,frame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.warn("deprecated", DeprecationWarning)
        frame_kp = mediapipe_functions.frame_process(model, frame)
    return frame_kp

def process_video(video_path, model, fps, flatten=False):
    keypoints_list = []
    frame_index = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to read the video feed from {video_path}")
        return None
    
    print(f"Processing video {video_path}")
    ret, frame = cap.read()

    while ret:
        keypoints = get_keypoints(model, frame)  # Shape: (2, N)
        keypoints = keypoints.T  # Shape: (N, 2)
        if flatten:
            keypoints = keypoints.flatten()  # Shape: (N*2,)
        keypoints_list.append(keypoints)
        ret, frame = cap.read()

        # frame_index += 1
        # if frame_index > 2:            # warning that is not finished
        #     warnings.warn("Stopping after 3 frames")
        #     break

    cap.release()

    keypoints_array = np.stack(keypoints_list) # Shape : (T, K, 2) or (T, K*2)

    return keypoints_array


def main():
    args = parse_arguments()
    FLATTEN = args.flatten
    FPS = args.fps
    VIDEO_DIR = args.video_dir

    # Initialize mediapipe model
    model = mediapipe_functions.model_init()

    # get data (video paths and labels)

    data = get_pucp_glosas_data(VIDEO_DIR)

    # Process each video

    for idx, row in data.iterrows():
        video_path = row['video_path']
        label = row['label']
        print(f"Processing video {video_path} with label {label}")

        keypoints_array = process_video(video_path, model, FPS, FLATTEN)
        # print(f"Keypoints array shape: {keypoints_array.shape}")
        # raise NotImplementedError("The following code is not implemented yet.")
        if keypoints_array is None:
            continue

        # save keypoints arrays
        os.makedirs('./keypoints', exist_ok=True)
        save_path = f'./keypoints/{label}.hdf5'

        with h5py.File(save_path, 'w') as h5_file:
            h5_file.create_dataset('keypoints', data=keypoints_array)
        
        #if idx >= 10:
          #  break
    mediapipe_functions.close_model(model)
    print("All videos have been processed.")

if __name__ == '__main__':
    main()
    