"""
@Project   : SMAAT Project
@File      : get_VT_labels_from_TG.py
@Author    : Ying Li
@Student ID: 20909226
@School    : EECMS
@University: Curtin University
@Date      : 2025-06-13
"""
"""
Brief module description:
This script scans participant folders for TextGrid files of isolated words, extracts GeMAPS
low-level descriptors from the corresponding WAVs using openSMILE, and saves per-file
feature arrays along with a JSON label file.

Purpose:
- Identify target word tokens (configured in WORDS) inside 'rebase_to_zero_TG' TextGrids.
- Determine whether each token contains a voicing transition (VT) error (VT_error_list).
- Extract GeMAPS features from the matching WAV in 'individual_wavs' and save as .npy.
- Append a label (1=VT error, 0=correct) to a consolidated labels JSON.

Requirements:
- Python packages: praatio, opensmile, pandas, numpy, pathlib
- Expected directory layout per participant:
    <participant>/
        rebase_to_zero_TG/   (TextGrid files)
        individual_wavs/     (matching WAV files)

Usage:
- Run the script directly. Adjust inpath/save_path and WORDS/VT_error_list as needed.
"""
import json
import pandas as pd
import numpy as np
import os
from praatio import textgrid
import opensmile
from pathlib import Path

WORDS = ['ba', 'eye', 'map', 'um', 'ham', 'papa', 'bob', 'pam', 'pup', 'pie', 'boy', 'bee', 'peep','bush', 'moon', 'phone', 'feet', 'fish', 'wash', 'show']
        #  'ten', 'dig', 'log', 'owl', 'cake', 'sun', 'snake', 'juice', 'clown', 'crib', 'grape',
        # 'cupcake', 'icecream', 'toothbrush', 'robot', 'banana', 'marshmallow', 'umbrella', 'hamburger', 'watermelon', 'rhinoceros']

def save_geMAPs_features_to_npy(geMAPs_features, output_path):
    """Save GeMAPs features to a .npy file."""
    
    np.save(output_path, geMAPs_features)
    print(f"GeMAPs features saved to {output_path}.")


def get_GeMAPs(wav_file):
    """Extract GeMAPs features from a given audio file."""
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    # dataframe
    y = smile.process_file(wav_file)
    # select the "Loudness_sma3", "mfcc1_sma3",	"mfcc2_sma3", "mfcc3_sma3",	"mfcc4_sma3", "f0semitoneFrom27.5Hz" features
    selected_features = ["Loudness_sma3", "mfcc1_sma3", "mfcc2_sma3", "mfcc3_sma3", "mfcc4_sma3", "F0semitoneFrom27.5Hz_sma3nz"]
    y = y[selected_features]
    # return dataframe
    return y
    
def get_labels_from_word_tier(tier):
    """Extract labels from the 'word' tier in the TextGrid."""
    # word_tier = tg.getTier("word")
    labels = []
    for start, end, label in tier.entries:
        label = label.split("/")[1] if '/' in label else label
        labels.append(label)
    return labels

if __name__ == "__main__":
    VT_error_list = ['FDV', 'IV', 'IDV', 'MDV', 'MV', '(+)NAS', 'DEN']
    save_path = "/mnt/data/ying/SMAAT_1st_iterative_learning/TD/Band_1"
    inpath = "/mnt/data/ying/SMAAT_1st_iterative_learning/TD/Band_1"
    all_counts = []
    labels_dict = {}
    for _f in os.listdir(inpath):
        parent_f = os.path.join(inpath, _f)
        input_directory = os.path.join(parent_f, 'rebase_to_zero_TG')
        
        # Skip if the directory does not exist
        if not os.path.isdir(input_directory):
            continue

        VT_error_files = 0
        VT_correct_files = 0
        other_error_files = 0
        
        for f in os.listdir(input_directory):
            info = f.rsplit('_', 1)[-1].rsplit('.', 1)[0]
            print(f"Processing file: {f} with info: {info}")
            if info in WORDS:
                tg_path = os.path.join(input_directory, f)
                try:
                    tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=True)
                except Exception as e:
                    print(f"Skipping file {f} due to TextGrid parsing error: {e}")
                    continue
                phonetic_tier = tg.getTier("phonetic")
                error_tier = tg.getTier("error")
                phonetic_label = get_labels_from_word_tier(phonetic_tier)
                error_label = get_labels_from_word_tier(error_tier)

                if len(phonetic_label) != 0:
                    valid_error_labels = [e for e in error_label if e.strip() != ""]
                    if valid_error_labels:
                        for error in valid_error_labels:
                            if error in VT_error_list:
                                VT_error_files += 1
                                print(f"VT Error label '{error}' found in file {f}.")
                                # replace the tg path to the wav path
                                wav_path = os.path.join(parent_f, 'individual_wavs', f.replace('.TextGrid', '.wav'))
                                if os.path.exists(wav_path):
                                    # Extract GeMAPs features
                                    ge_maps_df = get_GeMAPs(wav_path)
                                    # Save the GeMAPs features to a numpy file
                                    output_path = os.path.join(save_path, 'geMAPs_features_band1', f.replace('.TextGrid', '.npy'))
                                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                                    save_geMAPs_features_to_npy(ge_maps_df.to_numpy(), output_path)
                                    # add the filename and the error label to a labels.json file
                                    labels_dict[Path(output_path).name] = 1
                            else:
                                other_error_files += 1
                                print(f"Error label '{error}' is other errors in file {f}.")
                    else:
                        VT_correct_files += 1
                        print(f"No error label found in file {f}.")
                        # replace the tg path to the wav path
                        wav_path = os.path.join(parent_f, 'individual_wavs', f.replace('.TextGrid', '.wav'))
                        if os.path.exists(wav_path):
                            # Extract GeMAPs features
                            ge_maps_df = get_GeMAPs(wav_path)
                            # Save the GeMAPs features to a numpy file
                            output_path = os.path.join(save_path, 'geMAPs_features_band1', f.replace('.TextGrid', '.npy'))
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            save_geMAPs_features_to_npy(ge_maps_df.to_numpy(), output_path)
                            # add the filename and the error label to a labels.json file
                            labels_dict[Path(output_path).name] = 0  


        print(f"[{_f}] VT errors: {VT_error_files}, Correct: {VT_correct_files}, Other errors: {other_error_files}")
        
        counts = {
            "participant": _f,
            "VT_error_files": VT_error_files,
            "other_error_files": other_error_files,
            "VT_correct_files": VT_correct_files
        }
        all_counts.append(counts)
    
    # Save the labels to a JSON file
    labels_file = os.path.join(save_path, 'labels_band1.json')
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            master_labels = json.load(f)
    else:
        master_labels = {}
    master_labels.update(labels_dict)
    with open(labels_file, 'w') as f:
        json.dump(master_labels, f, indent=4)       
    
    print(f"Added {len(labels_dict)} new labels. Total: {len(master_labels)} saved to {labels_file}.")

    