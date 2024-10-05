import os
import shutil
import pandas as pd
from tqdm import tqdm

def copy_files( csv_file_path, new_folder_path, audio_base_folder):

    # Create new folder if it doesn't exist
    os.makedirs(new_folder_path, exist_ok=True)

    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Assuming the CSV file has a column named 'file_path' with relative or absolute paths to the audio files
    audio_paths = df['path']

    # Iterate through the file paths and copy each file to the new folder
    for audio_path in tqdm(audio_paths, desc="Copying audio files", unit="file"):
        full_audio_path = os.path.join(audio_base_folder, audio_path)  # Form the full path to the audio file
        if os.path.exists(full_audio_path):
            # Copy the file to the new folder
            shutil.copy(full_audio_path, new_folder_path)
            print(f"Copied: {full_audio_path}")
        else:
            print(f"File not found: {full_audio_path}")

    print("File copying completed!")

if __name__ == '__main__':

    csv_file_path = os.path.join('..', 'data', 'df_eval_19_3s.csv')
    new_folder_path = 'reduced_dataset'
    audio_base_folder = '/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_eval/flac'

    copy_files(csv_file_path, new_folder_path, audio_base_folder)