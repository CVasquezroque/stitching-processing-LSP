import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def save_statistics_to_txt(df, output_filename):
    with open(output_filename, 'w') as f:
        # Sentences statistics
        sentences = df['oracion'].unique()
        total_sentences = len(sentences)

        # Initialize counts
        sentences_with_transition_0_frames = 0
        sentences_with_transition_1_frame = 0
        sentences_with_all_transitions_0_frames = 0
        sentences_with_all_transitions_1_frame_or_less = 0
        sentence_with_any_transitions_more_than_1_frame = 0
        sentence_with_any_transitions_more_than_0_frame = 0
        for sentence in sentences:
            df_sentence = df[df['oracion'] == sentence]
            # Collect all transition frames columns
            tt_frames_cols = [col for col in df_sentence.columns if col.startswith('tt') and col.endswith('_frames')]
            # Get the frames as numpy array
            frames = df_sentence[tt_frames_cols].values.flatten()
            # Remove NaN values
            frames = frames[~np.isnan(frames)]
            # Convert to integer
            frames = frames.astype(int)

            # Check the conditions
            if any(frames == 0):
                sentences_with_transition_0_frames +=1
            if any(frames == 1):
                sentences_with_transition_1_frame +=1
            if all(frames == 0):
                sentences_with_all_transitions_0_frames +=1
            if all(frames <= 1):
                sentences_with_all_transitions_1_frame_or_less +=1
            if any(frames > 1):
                sentence_with_any_transitions_more_than_1_frame +=1
            if any(frames > 0):
                sentence_with_any_transitions_more_than_0_frame +=1

        # Similarly for n-grams
        total_ngrams = len(df)
        tt_frames_cols = [col for col in df.columns if col.startswith('tt') and col.endswith('_frames')]

        ngrams_with_transition_0_frames = df[df[tt_frames_cols].eq(0).any(axis=1)].shape[0]
        ngrams_with_transition_1_frame = df[df[tt_frames_cols].eq(1).any(axis=1)].shape[0]
        ngrams_with_all_transitions_0_frames = df[df[tt_frames_cols].eq(0).all(axis=1)].shape[0]
        ngrams_with_all_transitions_1_frame_or_less = df[df[tt_frames_cols].le(1).all(axis=1)].shape[0]
        ngrams_with_any_transitions_more_than_1_frame = df[df[tt_frames_cols].gt(1).any(axis=1)].shape[0]
        ngrams_with_any_transition_more_than_0_frame = df[df[tt_frames_cols].gt(0).any(axis=1)].shape[0]

        # Compute statistics for transitions with frames > 1
        # Get all transition frames and durations
        all_frames = df[tt_frames_cols].values.flatten()
        all_frames = all_frames[~np.isnan(all_frames)]
        all_frames = all_frames.astype(int)
        all_durations = []
        tt_duration_cols = [col for col in df.columns if col.startswith('tt') and col.endswith('_duration')]
        all_durations = df[tt_duration_cols].values.flatten()
        all_durations = all_durations[~np.isnan(all_durations)]

        # Filter transitions with frames > 1
        valid_indices = all_frames > 1
        filtered_frames = all_frames[valid_indices]
        filtered_durations = all_durations[valid_indices]

        # Compute statistics
        frames_stats = {
            'median': np.median(filtered_frames),
            'mean': np.mean(filtered_frames),
            'percentiles': np.percentile(filtered_frames, [25, 50, 75]),
            'max': np.max(filtered_frames),
            'min': np.min(filtered_frames)
        }
        durations_stats = {
            'median': np.median(filtered_durations),
            'mean': np.mean(filtered_durations),
            'percentiles': np.percentile(filtered_durations, [25, 50, 75]),
            'max': np.max(filtered_durations),
            'min': np.min(filtered_durations)
        }

        # Now write to the file
        f.write(f"Total number of sentences: {total_sentences}\n")
        f.write(f"Sentences with at least 1 transition with 0 frames: {sentences_with_transition_0_frames}, {sentences_with_transition_0_frames/total_sentences*100:.2f}%\n")
        f.write(f"Sentences with at least 1 transition with 1 frame: {sentences_with_transition_1_frame}, {sentences_with_transition_1_frame/total_sentences*100:.2f}%\n")
        f.write(f"Sentences with all transitions with 0 frames: {sentences_with_all_transitions_0_frames}, {sentences_with_all_transitions_0_frames/total_sentences*100:.2f}%\n")
        f.write(f"Sentences with all transitions with 1 frame or less: {sentences_with_all_transitions_1_frame_or_less}, {sentences_with_all_transitions_1_frame_or_less/total_sentences*100:.2f}%\n")
        f.write(f"Sentences with any transitions with more than 1 frame: {sentence_with_any_transitions_more_than_1_frame}, {sentence_with_any_transitions_more_than_1_frame/total_sentences*100:.2f}%\n")
        f.write(f"Sentences with any transitions with more than 0 frame: {sentence_with_any_transitions_more_than_0_frame}, {sentence_with_any_transitions_more_than_0_frame/total_sentences*100:.2f}%\n\n")

        f.write(f"Total number of n-grams: {total_ngrams}\n")
        f.write(f"n-grams with at least 1 transition with 0 frames: {ngrams_with_transition_0_frames}, {ngrams_with_transition_0_frames/total_ngrams*100:.2f}%\n")
        f.write(f"n-grams with at least 1 transition with 1 frame: {ngrams_with_transition_1_frame}, {ngrams_with_transition_1_frame/total_ngrams*100:.2f}%\n")
        f.write(f"n-grams with all transitions with 0 frames: {ngrams_with_all_transitions_0_frames}, {ngrams_with_all_transitions_0_frames/total_ngrams*100:.2f}%\n")
        f.write(f"n-grams with all transitions with 1 frame or less: {ngrams_with_all_transitions_1_frame_or_less}, {ngrams_with_all_transitions_1_frame_or_less/total_ngrams*100:.2f}%\n")
        f.write(f"n-grams with any transitions with more than 1 frame: {ngrams_with_any_transitions_more_than_1_frame}, {ngrams_with_any_transitions_more_than_1_frame/total_ngrams*100:.2f}%\n")
        f.write(f"n-grams with any transitions with more than 0 frame: {ngrams_with_any_transition_more_than_0_frame}, {ngrams_with_any_transition_more_than_0_frame/total_ngrams*100:.2f}%\n\n")

        f.write("Transition frames statistics (considering transitions with more than 1 frame):\n")
        f.write(f"Median: {frames_stats['median']}\n")
        f.write(f"Mean: {frames_stats['mean']}\n")
        f.write(f"25th percentile: {frames_stats['percentiles'][0]}\n")
        f.write(f"50th percentile: {frames_stats['percentiles'][1]}\n")
        f.write(f"75th percentile: {frames_stats['percentiles'][2]}\n")
        f.write(f"Max: {frames_stats['max']}\n")
        f.write(f"Min: {frames_stats['min']}\n\n")

        f.write("Transition durations statistics (seconds, considering transitions with more than 1 frame):\n")
        f.write(f"Median: {durations_stats['median']}\n")
        f.write(f"Mean: {durations_stats['mean']}\n")
        f.write(f"25th percentile: {durations_stats['percentiles'][0]}\n")
        f.write(f"50th percentile: {durations_stats['percentiles'][1]}\n")
        f.write(f"75th percentile: {durations_stats['percentiles'][2]}\n")
        f.write(f"Max: {durations_stats['max']}\n")
        f.write(f"Min: {durations_stats['min']}\n")
    print(f"Statistics have been saved to {output_filename}")


def generate_histograms(df):
    # Collect all transition frames and durations
    tt_frames_cols = [col for col in df.columns if col.startswith('tt') and col.endswith('_frames')]
    tt_duration_cols = [col for col in df.columns if col.startswith('tt') and col.endswith('_duration')]
    
    # Combine all transition frames into one list
    all_frames = []
    for col in tt_frames_cols:
        all_frames.extend(df[col])
    
    # Combine all transition durations into one list
    all_durations = []
    for col in tt_duration_cols:
        all_durations.extend(df[col])
    
    # Filter transitions with frames > 1
    filtered_frames = [frame for frame in all_frames if frame > 1]
    filtered_durations = [all_durations[idx] for idx, frame in enumerate(all_frames) if frame > 1]
    
    # Plot histogram for transition durations in seconds
    plt.figure(figsize=(10,6))
    sns.histplot(filtered_durations, bins=30, kde=False, color='skyblue', edgecolor='black')
    plt.title('Histogram of Transition Durations (seconds)')
    plt.xlabel('Duration (s)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('transition_durations_seconds_histogram.png', dpi=300)
    plt.close()
    print("Histogram of transition durations in seconds saved as 'transition_durations_seconds_histogram.png'")
    
    # Plot histogram for transition durations in frames
    plt.figure(figsize=(10,6))
    sns.histplot(filtered_frames, bins=range(int(min(filtered_frames)), int(max(filtered_frames))+1), kde=False, color='salmon', edgecolor='black')
    plt.title('Histogram of Transition Durations (frames)')
    plt.xlabel('Duration (frames)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('transition_durations_frames_histogram.png', dpi=300)
    plt.close()
    print("Histogram of transition durations in frames saved as 'transition_durations_frames_histogram.png'")

def generate_boxplots(df):
    # Collect all transition frames and durations
    tt_frames_cols = [col for col in df.columns if col.startswith('tt') and col.endswith('_frames')]
    tt_duration_cols = [col for col in df.columns if col.startswith('tt') and col.endswith('_duration')]

    # Combine all transition frames into one DataFrame
    frames_data = pd.melt(df[tt_frames_cols], value_name='Frames', var_name='Transition')
    durations_data = pd.melt(df[tt_duration_cols], value_name='Duration (s)', var_name='Transition')

    # Merge frames and durations
    transitions_data = pd.concat([frames_data['Frames'], durations_data['Duration (s)']], axis=1)

    # Filter transitions with frames > 1
    transitions_data = transitions_data[transitions_data['Frames'] > 1]

    # Set Seaborn style
    sns.set(style="whitegrid", context='paper', font_scale=1.2)

    # Boxplot for transition durations in seconds
    plt.figure(figsize=(8,6))
    sns.boxplot(x='Duration (s)', data=transitions_data, color='lightblue')
    plt.title('Boxplot of Transition Durations (seconds)')
    plt.ylabel('Duration (s)')
    plt.tight_layout()
    plt.savefig('transition_durations_seconds_boxplot.png', dpi=300)
    plt.close()
    print("Boxplot of transition durations in seconds saved as 'transition_durations_seconds_boxplot.png'")

    # Boxplot for transition durations in frames
    plt.figure(figsize=(8,6))
    sns.boxplot(x='Frames', data=transitions_data, color='lightgreen')
    plt.title('Boxplot of Transition Durations (frames)')
    plt.ylabel('Duration (frames)')
    plt.tight_layout()
    plt.savefig('transition_durations_frames_boxplot.png', dpi=300)
    plt.close()
    print("Boxplot of transition durations in frames saved as 'transition_durations_frames_boxplot.png'")
