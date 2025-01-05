# pucp_glosas_video_reader.py

import os
import pandas as pd

def get_pucp_glosas_data(root_path):
    data = []
    # root_path = 'E:/Home/Lab_Humanidades_Digitales/repositories/Datasets/PUCP_Glosas'  # Replace with the actual path to your videos
    for folder in os.listdir(root_path):
        if not folder.endswith('.xlsx'):
          for file in os.listdir(os.path.join(root_path, folder)):
              if file.endswith('.mp4') and 'ORACION' in file:
                  video_path = os.path.join(root_path, folder, file)
                  label = video_path.split('/')[-1][:-4]
                  data.append({'video_path': video_path, 'label': label})

    return pd.DataFrame(data)


if __name__ == '__main__':

    df = get_pucp_glosas_data(root_path='/home/shared/PUCP305/5. Segundo avance (corregido)/')
    print(df.head())
    print(f'Total videos: {len(df)}')
    print(f'Total labels: {len(df.label.unique())}')
    print(f'Video filename example: {df.video_path[0]}')
    print(f'Label example: {df.label[0]}')
