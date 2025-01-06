# -*- coding: utf-8 -*-
"""
Author: Carlos
Date: 2024-11-23
Last Update: 2024-11-26

Description: This script processes EAF files corresponding to sentences (ORACION). It generates 3-grams of glosses with their respective transitions.
Each 3-gram consists of three consecutive glosses and the durations of the transitions (tt1 and tt2) between them. The output is saved in a CSV file for further analysis.

Now, it also includes the start and end times for each gloss.

"""

import pympi
import argparse
import os
import pandas as pd
from utils.save_statistics import save_statistics_to_txt, generate_histograms, generate_boxplots # type: ignore

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate 3-grams of glosses with transitions from EAF files.')
    parser.add_argument('--rootPath', type=str, default='./', help='Root path where EAF files are located')
    parser.add_argument('--fileNameContains', type=str, default='ORACION', help='Substring that filenames must contain')
    parser.add_argument('--fps', type=float, default=30.0, help='Frames per second of the videos')
    parser.add_argument('--n', type=int, default=3, help='Number of glosses in each n-gram')
    args = parser.parse_args()
    return args

def find_eaf_files(root_path, file_name_contains):
    eaf_files = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith('.eaf') and file_name_contains in filename:
                file_path = os.path.join(dirpath, filename)
                eaf_files.append(file_path)
    return eaf_files

def process_eaf_file(eaf_file_path):
    aEAFfile = pympi.Elan.Eaf(eaf_file_path)
    eaf_file_name = os.path.basename(eaf_file_path)

    # Search for the gloss tier
    main_tier = 'GLOSA_IA'
    if main_tier not in aEAFfile.get_tier_names():
        print(f"Tier {main_tier} not found in {eaf_file_name}")
        return [], eaf_file_name
    
    # Extract gloss annotations

    dict_gloss = aEAFfile.get_annotation_data_for_tier(main_tier)
    annotations = []
    for annotation in dict_gloss:
        start_time = annotation[0] / 1000  # Convert to seconds
        end_time = annotation[1] / 1000      # Convert to seconds
        gloss_text = annotation[2]
        annotations.append({
            'gloss': gloss_text,
            'start_time': start_time,
            'end_time': end_time            
        })
    
    # Sort annotations by start time
    annotations.sort(key=lambda x: x['start_time'])
    
    return annotations, eaf_file_name

def generate_n_grams(n, annotations, eaf_file_name, fps):
    
    ngram_list = []
    num_annotations = len(annotations)

    for i in range(num_annotations - n + 1):
        glosses = [annotations[i + j] for j in range(n)]
        
        # Computer transitions between glosses
        transitions = []
        valid_ngram = True

        for j in range(n-1):
            transition_start = glosses[j]['end_time']
            transition_end = glosses[j+1]['start_time']
            transition_duration = transition_end - transition_start
            transition_frames = transition_duration * fps
            
            """
            Hay un error en la anotación con duración de transición de más de 3 segundos
            """
            # print(f'Sentence {eaf_file_name} -> {glosses[j]["gloss"]} -> {glosses[j+1]["gloss"]}: {transition_duration} seconds, {int(round(transition_frames))} frames') if transition_duration > 3 else None 
            # verify that transitions have positivo duration

            if transition_duration < 0 or transition_duration > 3:
                valid_ngram = False
                print(f'Invalid n-gram: {glosses[j]["gloss"]} -> {glosses[j+1]["gloss"]} in {eaf_file_name}')
                print(f'Invalid transition duration: {transition_duration} seconds')
                break
            
            transitions.append({
                'start_time': transition_start,
                'end_time': transition_end,
                'tt_duration': transition_duration,
                'tt_frames': int(round(transition_frames))
            })

        if not valid_ngram:
            continue

        # Build the n-gram

        ngram_record = {
            'oracion': eaf_file_name
        }
        # Add glosses and their times
        for idx, gloss in enumerate(glosses, 1):
            ngram_record[f'type{2 * idx - 1}'] = 'gloss'
            ngram_record[f'glosa{idx}'] = gloss['gloss']
            ngram_record[f'start_time{2 * idx - 1}'] = gloss['start_time']
            ngram_record[f'end_time{2 * idx - 1}'] = gloss['end_time']

            # If there's a transition after this gloss, add it
            if idx <= len(transitions):
                transition = transitions[idx - 1]
                ngram_record[f'type{2 * idx}'] = 'transition'
                ngram_record[f'start_time{2 * idx}'] = transition['start_time']
                ngram_record[f'end_time{2 * idx}'] = transition['end_time']
                ngram_record[f'tt{idx}_duration'] = transition['tt_duration']
                ngram_record[f'tt{idx}_frames'] = transition['tt_frames']

        ngram_list.append(ngram_record)


    return ngram_list

def save_ngrams_to_csv(ngrams_list, output_file):
    df_ngrams = pd.DataFrame(ngrams_list)
    df_ngrams.to_csv(output_file, index=False)
    print(f"n-grams have been saved to {output_file}")
    return df_ngrams

def main():
    args = parse_arguments()
    eaf_files = find_eaf_files(args.rootPath, args.fileNameContains)
    print(f"Found {len(eaf_files)} EAF files.")
    
    all_ngrams = []
    for eaf_file_path in eaf_files:
        annotations, eaf_file_name = process_eaf_file(eaf_file_path)
        if not annotations:
            continue  # Skip if no valid annotations
        ngrams_list = generate_n_grams(args.n, annotations, eaf_file_name, args.fps)
        all_ngrams.extend(ngrams_list)
    
    if all_ngrams:
        # Save all n-grams to a CSV file
        output_file = f'{args.n}grams_with_transitions.csv'
        df_ngrams = save_ngrams_to_csv(all_ngrams, output_file)

        # Generate histograms
        generate_histograms(df_ngrams)

        # Save statistics
        output_txt = f'{args.n}grams_with_transitions_statistics.txt'
        save_statistics_to_txt(df_ngrams, output_txt)
        generate_boxplots(df_ngrams)
        generate_histograms(df_ngrams)      
    else:
        print("No n-grams were generated.")

if __name__ == '__main__':
    main()