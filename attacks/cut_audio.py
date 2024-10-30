import os
import librosa
import soundfile as sf
from tqdm import tqdm

def cut_files(folder_clean, folder_pert, out_folder):
    # Define paths
    folder_A = folder_pert
    folder_B = folder_clean
    output_folder = out_folder

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get lists of files in each folder
    files_A = [f for f in os.listdir(folder_A) if f.endswith('.flac')]
    files_B = [f for f in os.listdir(folder_B) if f.endswith('.flac')]

    # Create a dictionary for files in folder B with the format {number: filepath}
    files_B_dict = {}
    for file in files_B:
        # Extract the number from filenames in folder B (e.g., "2834763" from "LA_E_2834763.flac")
        number = file.split('_')[-1].split('.')[0]
        files_B_dict[number] = os.path.join(folder_B, file)

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
            if len(audio_B) < len(audio_A):
                # Trim audio_A to the length of audio_B
                trimmed_audio = audio_A[:len(audio_B)]
            else:
                # No trimming needed, use original audio
                trimmed_audio = audio_A

            # Save the trimmed audio to the output folder with the same name as in folder A
            output_path = os.path.join(output_folder, file_A)
            sf.write(output_path, trimmed_audio, sr_A)



if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script
    folder_clean = '/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_eval/flac'
    folder_pert = os.path.join(script_dir, 'Ens1D_ResSEN_v0_pow/Ens1D_ResSEN_v0_whole_pow_30_50_0dot009_0dot002_X')
    folder_out = os.path.join(script_dir, 'Ens1D_ResSEN_v0_pow/Ens1D_ResSEN_v0_whole_pow_30_50_0dot009_0dot002')

    cut_files(folder_clean, folder_pert, folder_out)