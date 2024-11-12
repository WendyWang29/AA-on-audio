import os
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np


def compute_snr(clean_signal, perturbed_signal):
    # Ensure both signals are of the same length
    if len(clean_signal) != len(perturbed_signal):
        raise ValueError("Signals must be of the same length to compute SNR.")

    # Calculate noise (difference between perturbed and clean signals)
    noise = perturbed_signal - clean_signal

    # Compute signal power and noise power
    signal_power = np.sum(clean_signal ** 2)
    noise_power = np.sum(noise ** 2)

    # Avoid division by zero
    if noise_power == 0:
        return float('inf')  # Perfectly identical signals have infinite SNR

    # Compute SNR in dB
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def avg_snr(folder_clean, folder_pert):

    # Define paths
    folder_A = folder_pert
    folder_B = folder_clean

    # Get lists of files in each folder
    files_A = [f for f in os.listdir(folder_A) if f.endswith('.flac')]
    files_B = [f for f in os.listdir(folder_B) if f.endswith('.flac')]

    # Create a dictionary for files in folder B with the format {number: filepath}
    files_B_dict = {}
    for file in files_B:
        # Extract the number from filenames in folder B (e.g., "2834763" from "LA_E_2834763.flac")
        number = file.split('_')[-1].split('.')[0]
        files_B_dict[number] = os.path.join(folder_B, file)

    snr_values = []  # List to store SNR values for each file

    # Process each file in folder A
    for file_A in tqdm(files_A, desc="Processing files from folder A"):
        # Extract the number from filenames in folder A (e.g., "1000147" from "Ens1D_RawSEN_20_60_v0_pow_LA_E_1000147_None.flac")
        number = file_A.split('_')[-2]

        # Check if there is a corresponding file in folder B
        if number in files_B_dict:
            # Load files from folder A and folder B
            file_A_path = os.path.join(folder_A, file_A)
            file_B_path = files_B_dict[number]

            # Load the audio files
            audio_A, sr_A = librosa.load(file_A_path, sr=None)
            audio_B, sr_B = librosa.load(file_B_path, sr=None)

            # Ensure both files have the same sample rate
            if sr_A != sr_B:
                print(f"Sample rate mismatch for {file_A} and {files_B_dict[number]}, skipping...")
                continue

            # Check lengths
            if len(audio_A) < len(audio_B):
                # Trim audio_B to the length of audio_A
                trimmed_audio_B = audio_B[:len(audio_A)]
            else:
                # No trimming needed, use original audio
                trimmed_audio_B = audio_B

            # Compute and store SNR
            snr_value = compute_snr(trimmed_audio_B, audio_A)
            snr_values.append(snr_value)

    # Calculate and print the average SNR
    if snr_values:
        average_snr = np.mean(snr_values)
        print(f"Average SNR: {average_snr:.2f} dB")
    else:
        print("No SNR values computed. Check the input files and paths.")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script
    folder_clean = '/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_eval/flac'
    folder_pert = os.path.join(script_dir, 'Ens1D_RawSEN_v0_pow/Ens1D_RawSEN_v0_whole_pow_20_60_0dot005_0dot002')

    print(f'working on {folder_pert}')

    avg_snr(folder_clean, folder_pert)