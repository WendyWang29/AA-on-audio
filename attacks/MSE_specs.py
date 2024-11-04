import os
import numpy as np
import librosa
from tqdm import tqdm


def compute_mse(a, b):
    return np.mean((a - b) ** 2)

def MSE_specs(folder_A, folder_B):
    # folder A --> attacked file
    # folder B --> clean files

    # Parameters for STFT
    win_length = 2048
    n_fft = 2048
    hop_length = 512
    window = 'hann'

    # Store MSE values
    mse_values = []

    # Get the list of files in folder A
    files_A = os.listdir(folder_A)

    # Loop through each file in folder A
    for file_A in tqdm(files_A):
        # Extract the unique identifier from the filename, if possible
        parts = file_A.split('_')
        identifier = parts[-2] if len(parts) >= 2 else None

        # Skip files that don't have a valid identifier
        if identifier is None:
            print(f"Skipping file {file_A} as it does not contain a valid identifier.")
            continue

        # Construct the expected filename in folder B
        file_B = f'LA_E_{identifier}.flac'
        file_B_path = os.path.join(folder_B, file_B)

        # Check if the corresponding file exists in folder B
        if os.path.isfile(file_B_path):
            # Load audio files
            audio_A, sr_A = librosa.load(os.path.join(folder_A, file_A), sr=None)
            audio_B, sr_B = librosa.load(file_B_path, sr=None)
            audio_B = audio_B[:47104]

            # Compute the power spectrogram for file A
            s_A = librosa.stft(audio_A, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
            pow_spec_A = librosa.power_to_db(np.abs(s_A) ** 2, ref=np.max)

            # Compute the power spectrogram for file B
            s_B = librosa.stft(audio_B, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
            pow_spec_B = librosa.power_to_db(np.abs(s_B) ** 2, ref=np.max)

            # Compute MSE between the two power spectrograms
            mse = compute_mse(pow_spec_A, pow_spec_B)
            mse_values.append(mse)
        else:
            print(f"File {file_B} not found in folder B.")

    # Compute the average MSE if there are valid values
    if mse_values:
        average_mse = np.mean(mse_values)
        print(f"Average MSE: {average_mse}")
    else:
        print("No valid MSE values to compute an average.")



if __name__ == '__main__':
    file_number = 1001893
    attack = 'NoAttack'
    attack_model = 'ResNet2D'
    epsilon = None
    script_dir = os.path.dirname(__file__)
    folder_A = os.path.join(script_dir, f'{attack}_{attack_model}_v0_pow',
                            f'{attack}_{attack_model}_v0_whole_pow_{epsilon}')
    folder_B = clean_audio_path = os.path.join(script_dir,
                                               '..', f'/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_eval/flac')
    MSE_specs(folder_A, folder_B)