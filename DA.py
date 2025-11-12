'''
@Project   : SMAAT Project
@File      : DA.py
@Author    : Ying Li
@Student ID: 20909226
@School    : EECMS
@University: Curtin University
'''

"""
Data augmentation utilities for the SMAAT project.

This script augments minority-class audio files by applying dynamic volume
mapping and time-stretch transformations, saves augmented WAVs, extracts and
stores geMAPS features (via get_VT_labels_from_TG), and updates a labels JSON.
Designed to be run as a standalone script that reads a labels JSON, selects
files to augment, and writes augmented audio + feature files into the dataset
folder structure.

Contains: get_augment, adjust_dynamic_volume, get_word_count, save_wordcounts_in_excel,
and a __main__ workflow to generate and register augmentations.
"""

import librosa
import numpy as np
import json
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf
import get_VT_labels_from_TG
import os
import pandas as pd
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import random

def get_augment():
    return Compose([
        # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, leave_length_unchanged=False, p=1.0),
        # PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        # Shift(p=0.5),
    ])

def adjust_volume(audio, sr, change_db):
    """ Adjust the volume of audio data
        audio: numpy array of audio
        sr: sample rate of audio
        change_db: decibel change (+ increase, - decrease)
    """
    audio = librosa.db_to_amplitude(change_db) * audio
    return audio

def get_word_count(selected_files):
    """ Get the word count from a json file
        json_path: path to the json file
    """
    word_counts = {}
    for f in selected_files:
        word = Path(f).stem.split('_')[-1]  # Assuming the word is the last part of the filename
        if word not in word_counts:
            word_counts[word] = 0
        word_counts[word] += 1
    
    return word_counts
    
def save_wordcounts_in_excel(word_counts):
    # Save word counts to Excel
    word_counts_df = pd.DataFrame(list(word_counts.items()), columns=['Word', 'Count'])
    excel_path = '/home/ying/preprocess_SMAAT/voicing_transitions/word_counts.xlsx'
    word_counts_df.to_excel(excel_path, index=False)
    print(f"Word counts saved to {excel_path}")
    
if __name__ == "__main__":
    # Read a JSON file
    json_path = '/mnt/data/ying/SMAAT_1st_iterative_learning/TD/Band_3/labels_band3.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(len(data), "files in the labels_combined JSON file")
    
    # Select all files with label 1
    selected_files = [fname for fname, label in data.items() if label == 1 and "aug" not in fname]
    print("Selected files with label 1:", selected_files, len(selected_files))
    word_counts = get_word_count(selected_files)
    print(f"Word counts: {word_counts}")
    
    num_majority_files = len(data) - len(selected_files)
    num_to_sample = num_majority_files - len(selected_files)
    print("Number of files to sample from majority class:", num_to_sample)
    labels_dict = {}
    inpath = "/mnt/data/ying/SMAAT_1st_iterative_learning/TD/Band_3"
    for file in selected_files:
        filename = Path(file).stem
        print("Processing file:", filename)
        participant_ID = filename.split('_A0')[0]
        print("Participant ID:", participant_ID)
        audio_path = f"{inpath}/{participant_ID}/individual_wavs/{filename}.wav"
        print("Audio path:", audio_path)
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)

        word = Path(file).stem.split('_')[-1]
        num_augment = num_to_sample// len(word_counts) // word_counts[word]
        remainder = num_to_sample // len(word_counts) % word_counts[word]
        print(f"Number of augmentations for {filename}: {num_augment} with remainder {remainder}")

        # For each selected file, augment enough times using the get_augment function
        augment = get_augment()
        for i in range(num_augment + (1 if remainder >= 1 else 0)):
            # Generate a random volume change between -3 dB and +3 dB
            change_db = np.random.uniform(-3, 3)
            # Adjust the volume first
            y_vol, gamma = adjust_volume(y, sr, change_db)
            
            # Generate a random volume change between -3 dB and +3 dB
            # change_db = np.random.uniform(-3, 3)
            # Adjust the volume
            augmented_audio = get_augment()
            # Save the adjusted audio
            out_dir = Path(inpath) / "individual_wavs_audiomentations"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{filename}_aug{i}_vol{change_db:.2f}.wav"
            sf.write(str(out_path), augmented_audio, sr)
            print(f"Saved adjusted audio to {out_path}")
            ge_maps_df = get_VT_labels_from_TG.get_GeMAPs(str(out_path))
            fname = f"{Path(out_path).stem}.npy"
            output_path = os.path.join(inpath, 'geMAPs_features_audiomentations', fname)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            get_VT_labels_from_TG.save_geMAPs_features_to_npy(ge_maps_df.to_numpy(), output_path)
            # add the filename and the error label to a labels.json file
            labels_dict[Path(output_path).name] = 1
            # # Generate random parameters for each augmentation
            # # noise_min = np.random.uniform(0.001, 0.01)
            # # noise_max = np.random.uniform(noise_min, 0.015)
            # stretch_min = np.random.uniform(0.8, 1.0)
            # stretch_max = np.random.uniform(stretch_min, 1.25)
            # # pitch_min = np.random.uniform(-4, 0)
            # # pitch_max = np.random.uniform(pitch_min, 4)

            # # Create a new augment pipeline with these random parameters
            # augment = Compose([
            #     # AddGaussianNoise(min_amplitude=noise_min, max_amplitude=noise_max, p=0.5),
            #     TimeStretch(min_rate=stretch_min, max_rate=stretch_max, leave_length_unchanged=False, p=1.0),
            #     # PitchShift(min_semitones=pitch_min, max_semitones=pitch_max, p=0.5),
            #     # Shift(p=0.5),
            # ])
            # # Then apply augmentation (time stretch)
            # augmented_samples = augment(samples=y_vol, sample_rate=sr)
            # output_dir = Path(inpath) / "individual_wavs_audiomentations_dynamic_volume_timestretch"
            # output_dir.mkdir(parents=True, exist_ok=True)
            # # Create a unique filename for the augmented audio
            # params_str = f"gamma_{gamma}_stretch_{stretch_min:.2f}-{stretch_max:.2f}"

            # output_filename = f"{filename}_aug{i}_{params_str}.wav"
            # output_path = output_dir / output_filename
            # sf.write(str(output_path), augmented_samples, sr)
            # print(f"Augmented audio saved to {output_path}")   
            # ge_maps_df = get_VT_labels_from_TG.get_GeMAPs(str(output_path))
            # fname = f"{Path(output_path).stem}.npy"
            # output_path = os.path.join(inpath, 'geMAPs_features_audiomentations_dynamic_volume_timestretch', fname)
            # os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # get_VT_labels_from_TG.save_geMAPs_features_to_npy(ge_maps_df.to_numpy(), output_path)
            # # add the filename and the error label to a labels.json file
            # labels_dict[Path(output_path).name] = 1 
            
    # Save the labels to a JSON file
    labels_file = os.path.join(inpath, 'labels_DA_audiomentations_dynamic_volume_timestretch.json')
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            master_labels = json.load(f)
    else:
        master_labels = {}
    master_labels.update(labels_dict)
    with open(labels_file, 'w') as f:
        json.dump(master_labels, f, indent=4)       
    
    print(f"Added {len(labels_dict)} new labels. Total: {len(master_labels)} saved to {labels_file}.")