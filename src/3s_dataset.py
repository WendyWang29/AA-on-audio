import librosa
import pandas as pd
import os
from tqdm import tqdm

def get_audio_dur(file_path):
    try:
        audio, sr = librosa.load(file_path)
        return librosa.get_duration(y=audio, sr=sr)
    except Exception as e:
        print(f'Error loading {file_path}: {e}')
        return 0


if __name__ == '__main__':
    input_csv = '../data/df_eval_19.csv'
    output_csv_path = '../data/df_eval_19_3s.csv'
    min_dur = 3.0

    # make df_eval_19 read only
    os.chmod(input_csv, 0o444)
    print(f'{input_csv} is now read only\n')

    df = pd.read_csv(input_csv)

    filtered_rows = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Processing audio files...'):
        file_path = row['path']
        duration = get_audio_dur(file_path)
        if duration > min_dur:
            filtered_rows.append(row)

    filtered_df = pd.DataFrame(filtered_rows)
    filtered_df.to_csv(output_csv_path, index=False)  # keep only the original indices from df_eval_19.csv

    print('Done :D')



