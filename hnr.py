'''
@Project   : SMAAT Project
@File      : hnr.py
@Author    : Ying Li
@Student ID: 20909226
@School    : EECMS
@University: Curtin University
'''

'''
Function to determine if a frame is voiced based on ZCR, HNR, energy, and pitch
Note: Adjust thresholds based on your specific requirements and data characteristics
This function returns True if the frame is considered voiced, otherwise False. This function was dropped because it was not accurate.
pseudocode:
def is_voiced(zcr, hnr, energy, pitch):
    score = 0
    if zcr < 0.07: score += 1
    if hnr > 5: score += 1
    if energy > -30: score += 1
    if 60 <= pitch <= 400: score += 1
    
    return score >= 3
'''

import parselmouth
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d


def is_voiced(zcr, hnr, energy, pitch):
    score = 0
    if zcr < 0.1:
        score += 1
    if hnr > 5:
        score += 1
    if energy > 40:
        score += 1
    if pitch > 200 and pitch < 500:
        score += 1
    return score >= 3


if __name__ == "__main__":
    # Load audio
    # filename = "/home/ying/preprocess_SMAAT/voicing_transitions/hnr_res/VT_corrects/New_Recording_398.wav"  
    filename = "/mnt/data/ying/SMAAT_1st_iterative_learning/TD/Band_3/363_Nicholas/individual_wavs/363_Nicholas_A062_05281536_C021_fish.wav"
    # filename = "/mnt/data/ying/SMAAT_1st_iterative_learning/TD/Band_3/428_Alex/individual_wavs/428_Alex_A062_05201509_C021_papa.wav"
    snd = parselmouth.Sound(filename)
    y, sr = librosa.load(filename, sr=None)

    # Parameters
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    #  ZCR Calculation (librosa) 
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr_times = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=hop_length)

    #  HNR Calculation (parselmouth) 
    harmonicity = snd.to_harmonicity_cc(time_step=0.010, minimum_pitch=75)
    hnr_times = []
    hnr_values = []
    t = harmonicity.xmin
    while t <= harmonicity.xmax:
        hnr = harmonicity.get_value(t)
        if hnr is not None:
            hnr_times.append(t)
            hnr_values.append(hnr)
        t += 0.010
    #  Pitch (F0) Extraction 
    pitch = snd.to_pitch_ac(time_step=0.010, pitch_floor=75, pitch_ceiling=700)
    pitch_times = []
    pitch_values = []
    t = pitch.xmin
    while t <= pitch.xmax:
        f0 = pitch.get_value_at_time(t)
        if f0 is not None and f0 > 0:
            pitch_times.append(t)
            pitch_values.append(f0)
        t += 0.010
    
    # Extract intensity (in dB)
    intensity_obj = snd.to_intensity(time_step=0.01)  # 10 ms step (adjustable)
    # Get time and intensity values
    times = intensity_obj.xs()
    intensities_db = intensity_obj.values[0]  # in decibels (dB)

    

    # #  Pulse Extraction 
    # pitch = snd.to_pitch(pitch_floor=75, pitch_ceiling=500)
    # pulses = parselmouth.praat.call([snd, pitch], "To PointProcess (cc)")
    # print(pulses)
    # num_pulses = pulses.get_number_of_points()
    # pulse_times = [pulses.get_time_of_point(i + 1) for i in range(num_pulses)]

    #  Plotting with triple y-axis 
    fig, ax1 = plt.subplots(figsize=(12, 6))
    # Plot HNR, ZCR, Pitch, and Intensity on the same figure with different y-axes
    # Set fixed y-axis limits for all axes to ensure consistent intervals across samples
    hnr_ylim = (-250, 40)         # Example: HNR in dB
    zcr_ylim = (0, 0.4)          # Example: ZCR (unitless, usually 0-0.4 for speech)
    pitch_ylim = (75, 700)       # Example: Pitch in Hz
    intensity_ylim = (20, 70)    # Example: Intensity in dB

    ax1.set_ylim(hnr_ylim)
    # ax2, ax3, ax4 will be created below and also set their y-limits
    # HNR (left y-axis)
    ax1.plot(hnr_times, hnr_values, label="HNR (dB)", color="tab:blue")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("HNR (dB)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # # Pulses
    # for pt in pulse_times:
        # ax1.axvline(pt, color='red', linestyle='--', linewidth=0.4, alpha=0.6)

    # ZCR (first right y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylim(zcr_ylim)
    ax2.plot(zcr_times, zcr, label="ZCR", color="tab:green", alpha=0.7)
    ax2.set_ylabel("ZCR", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    # Pitch (second right y-axis, offset)
    ax3 = ax1.twinx()
    ax3.set_ylim(pitch_ylim)
    ax3.spines["right"].set_position(("axes", 1.1))  # shift 10% to the right
    ax3.plot(pitch_times, pitch_values, label="Pitch (Hz)", color="orange", linestyle="--")
    ax3.set_ylabel("Pitch (Hz)", color="orange")
    ax3.tick_params(axis="y", labelcolor="orange")


    # Intensity (third right y-axis, offset)
    ax4 = ax1.twinx()
    ax4.set_ylim(intensity_ylim)
    ax4.spines["right"].set_position(("axes", 1.2))
    ax4.plot(times, intensities_db, label="Intensity (dB)", color="purple", linestyle=":")
    ax4.set_ylabel("Intensity (dB)", color="purple")
    ax4.tick_params(axis="y", labelcolor="purple")  

    #  Voicing Decision for Each Frame 
    # Interpolate all features to a common time base (e.g., intensity times)

    common_times = times  # Use intensity time stamps as reference

    # Interpolate ZCR
    zcr_interp = interp1d(zcr_times, zcr, bounds_error=False, fill_value="extrapolate")
    zcr_on_common = zcr_interp(common_times)

    # Interpolate HNR
    hnr_interp = interp1d(hnr_times, hnr_values, bounds_error=False, fill_value="extrapolate")
    hnr_on_common = hnr_interp(common_times)

    # Interpolate Pitch
    pitch_interp = interp1d(pitch_times, pitch_values, bounds_error=False, fill_value=0)
    pitch_on_common = pitch_interp(common_times)

    # Now, for each frame, decide voiced/voiceless
    voicing_decisions = []
    for z, h, e, p in zip(zcr_on_common, hnr_on_common, intensities_db, pitch_on_common):
        voiced = is_voiced(z, h, e, p)
        voicing_decisions.append(voiced)
    # Optionally, plot voicing decisions as a background color
    for idx, (t, v) in enumerate(zip(common_times, voicing_decisions)):
        if v:
            ax1.axvspan(t, t + 0.01, color='yellow', alpha=0.15, zorder=0)

    # Final styling
    fig.suptitle("HNR, ZCR, Pitch and Intensity Over Time (363_Nicholas [fish])", fontsize=14)
    fig.tight_layout()
    plt.grid(True)
    fig.subplots_adjust(top=0.92)
    

    # Save to PDF
    pdf_path = "/home/ying/preprocess_SMAAT/voicing_transitions/hnr_res/meeting_with_Neville/correct_363_Nicholas_A062_05281536_C021_fish.pdf"
    # pdf_path = "/home/ying/preprocess_SMAAT/voicing_transitions/hnr_res/meeting_with_Neville/IV_384_Joshua_A062_09171350_C023_pam.pdf"
    # pdf_path = "/home/ying/preprocess_SMAAT/voicing_transitions/hnr_res/VT_corrects/428_Alex_A062_05201509_C021_papa_voicing_detection_output_test.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        plt.close()

    print(f"Plot saved to: {pdf_path}")
    