"""
Various tests to study
* how the training and the evaluation work
* evaluate one single .flac file from the evaluation list
* plot the spectrogram of one single .flac file from the evaluation list

author: wwang
"""

import librosa

from src.resnet_model import SpectrogramModel
from src.resnet_utils import get_features
from src.utils import *
import numpy as np
import matplotlib.pyplot as plt
import logging
from src.audio_utils import read_audio

def load_spec_model(device, config):
    """
    Load the spectrogram model - pre-trained
    :param device: GPU or CPU
    :param config: config file path
    :return: model
    """
    resnet_spec_model = SpectrogramModel().to(device)
    resnet_spec_model.load_state_dict(torch.load(config['model_path_spec'], map_location=device))
    return resnet_spec_model

def plot_spec(spec, file):
    n_fft = 2048
    hop_length = 512
    sr = 16000
    frame_duration = hop_length / sr

    # compute time value for each frame
    frame_times = np.linspace(0,spec.shape[1]-1, 5) * frame_duration
    frame_times = [f'{value:.3f}' for value in frame_times]

    # compute frequency values for each freq bin
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft) #1025 values
    num_y_ticks = 10
    y_ticks_idx = np.linspace(0, len(freqs)-1, num_y_ticks, dtype=int)

    plt.figure(figsize=(10, 6))
    plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')

    plt.xlabel('Time (s)')
    plt.xticks(np.linspace(0, spec.shape[1]-1, 5), frame_times)

    plt.yticks(y_ticks_idx, [f'{int(freqs[idx])} Hz' for idx in y_ticks_idx])


    plt.colorbar(label='Intensity (dB)')
    file_name_for_title = file.split('/')[-3:]
    file_name_for_title = '/'.join(file_name_for_title)
    plt.title(f'Power spectrogram for file\n {file_name_for_title} ')
    plt.show()

def eval_one_file_spec(index, config):
    """
    Evaluate the spectrogram model on one single file taken from eval list
    :param index: the index of the single file you want to evaluate
    :param config: config file
    :return: predictions as a tensor
    """

    # read the .csv and get the list of paths to the files
    df_eval = pd.read_csv(config['df_eval_path'])
    file_eval = list(df_eval['path'])

    # get the single file path
    file = file_eval[index]
    print(f'\n Evaluating file {file}')

    # get the feature (the cached spec is a ndarray: (1025,41) )
    X = get_features(wav_path=file,
                     features=config['features'],
                     args=config,
                     X=None,
                     cached=True,
                     force=False)

    # plot the spectrogram
    plot_spec(X, file)

    # transform into a mini batch and to a tensor
    X_batch = np.expand_dims(X, axis=0)  # shape ndarray (1,1025,41)
    X_batch_tensor = torch.from_numpy(X_batch).to(device) # tensor (1,1025,41)

    # get the prediction
    pred = model(X_batch_tensor)

    return pred


if __name__ == '__main__':

    # pre ###########################
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    seed_everything(1234)
    set_gpu(-1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ##################################

    # get the config settings
    config_path = '../config/residualnet_train_config.yaml'
    config_res = read_yaml(config_path)

    # load the spectrogram model
    model = load_spec_model(device, config_res)
    model.eval()

    # evaluate one single file and show the spectrogram
    pred = eval_one_file_spec(index=2, config=config_res)
    print(f'\n The predictions for the file are {pred} \n the score is {pred[0][0]-pred[0][1]}')
