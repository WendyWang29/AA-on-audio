"""
Various tests to study...
* how the training and the evaluation work
* evaluate one single .flac file from the evaluation list
* plot the spectrogram of one single .flac file from the evaluation list

author: wwang
"""

import librosa
import scipy.signal.windows as windows
import math
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



def recover_mag_spec(power_spec):
    # recover linear power spec from dB
    power_spec_linear = librosa.db_to_power(power_spec)
    # take sqrt to recover magnitude
    magnitude_spec = np.sqrt(power_spec_linear)
    return magnitude_spec


"""
##### SPSI algorithm #####
for the details check the appendix below
implementation taken from https://github.com/lonce/SPSI_Python/blob/master/SPSI_notebook/spsi.ipynb

inputs:
    msgram = magnitude spectrogram ([frequencies, frames])
    fftsize = window length
    hop_size

returns:
    y = audio signal
"""
def spsi(msgram, n_fft, hop_length):
    numBins, numFrames = msgram.shape
    y_out = np.zeros(numFrames * hop_length + n_fft - hop_length)

    m_phase = np.zeros(numBins);
    m_win = windows.hann(n_fft, sym=True)  # assumption here that hann was used to create the frames of the spectrogram

    # processes one frame of audio at a time
    for i in range(numFrames):
        m_mag = msgram[:, i]
        for j in range(1, numBins - 1):
            if (m_mag[j] > m_mag[j - 1] and m_mag[j] > m_mag[j + 1]):  # if j is a peak
                alpha = m_mag[j - 1];
                beta = m_mag[j];
                gamma = m_mag[j + 1];
                denom = alpha - 2 * beta + gamma;

                if (denom != 0):
                    p = 0.5 * (alpha - gamma) / denom;
                else:
                    p = 0;

                # phaseRate=2*math.pi*(j-1+p)/fftsize;                  # adjusted phase rate
                phaseRate = 2 * math.pi * (j + p) / n_fft;  # adjusted phase rate
                m_phase[j] = m_phase[j] + hop_length * phaseRate;  # phase accumulator for this peak bin
                peakPhase = m_phase[j];

                # If actual peak is to the right of the bin freq
                if (p > 0):
                    # First bin to right has pi shift
                    bin = j + 1;
                    m_phase[bin] = peakPhase + math.pi;

                    # Bins to left have shift of pi
                    bin = j - 1;
                    while ((bin > 1) and (m_mag[bin] < m_mag[bin + 1])):  # until you reach the trough
                        m_phase[bin] = peakPhase + math.pi;
                        bin = bin - 1;

                    # Bins to the right (beyond the first) have 0 shift
                    bin = j + 2;
                    while ((bin < (numBins)) and (m_mag[bin] < m_mag[bin - 1])):
                        m_phase[bin] = peakPhase;
                        bin = bin + 1;

                # If actual peak is to the left of the bin frequency
                if (p < 0):
                    # First bin to left has pi shift
                    bin = j - 1;
                    m_phase[bin] = peakPhase + math.pi;

                    # and bins to the right of me - here I am stuck in the middle with you
                    bin = j + 1;
                    while ((bin < (numBins)) and (m_mag[bin] < m_mag[bin - 1])):
                        m_phase[bin] = peakPhase + math.pi;
                        bin = bin + 1;

                    # and further to the left have zero shift
                    bin = j - 2;
                    while ((bin > 1) and (m_mag[bin] < m_mag[bin + 1])):  # until trough
                        m_phase[bin] = peakPhase;
                        bin = bin - 1;

            # end ops for peaks
        # end loop over fft bins with

        magphase = m_mag * np.exp(1j * m_phase)  # reconstruct with new phase (elementwise mult)
        magphase[0] = 0;
        magphase[numBins - 1] = 0  # remove dc and nyquist
        m_recon = np.concatenate([magphase,
                                  np.flip(np.conjugate(magphase[1:numBins - 1]),
                                          0)])

        # overlap and add
        m_recon = np.real(np.fft.ifft(m_recon)) * m_win
        y_out[i * hop_length:i * hop_length + n_fft] += m_recon

    return y_out


def griffin_lim(magnitude_spectrogram, n_fft, hop_length, init_phase, num_iterations=100):
    """
    Griffin-Lim phase reconstruction algorithm.

    Parameters:
        magnitude_spectrogram (2D numpy array): Magnitude spectrogram of the signal.
        n_fft (int): Size of the FFT window.
        hop_length (int): Number of samples between successive frames.
        num_iterations (int): Number of iterations for the phase reconstruction (default: 100).

    Returns:
        np.array: Time-domain signal reconstructed from the magnitude spectrogram.
    """
    phase = init_phase
    for i in range(num_iterations):
        # Reconstruct complex spectrogram
        complex_spectrogram = magnitude_spectrogram * phase

        # Reconstruct time-domain signal via inverse STFT
        audio_signal = librosa.istft(complex_spectrogram, hop_length=hop_length, win_length=n_fft)

        # Compute STFT of reconstructed signal to obtain phase
        complex_spectrogram_new = librosa.stft(audio_signal, n_fft=n_fft, hop_length=hop_length)
        phase = complex_spectrogram_new / np.maximum(1e-8, np.abs(complex_spectrogram_new))

    return audio_signal



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
