"""
SIGNAL PROCESSING UTILS
#TODO delete from tests.tests.py
"""
import librosa
import numpy as np
import os
import pandas as pd
import math
import scipy.signal.windows as windows
from src.audio_utils import read_audio
from src.resnet_features import compute_spectrum
from src.rawnet_utils import create_mini_batch_RawNet

def get_spectrogram_from_audio(audio_path):
    audio, fs = read_audio(audio_path,
                           dur=180,
                           fs=16000,
                           trim=False,
                           int_type=False,
                           windowing=False)

    spec = compute_spectrum(audio)
    return spec

def retrieve_single_audio(config, index):
    df_eval = pd.read_csv(os.path.join('..', config["df_eval_path"]))

    # get list of all eval files and labels
    file_eval = list(df_eval['path'])
    label_eval = list(df_eval['label'])

    # get one single file and its label given an index
    file = file_eval[index]
    X, fs = read_audio(file)
    # print(f'File {0} is {X.size} samples long with fs={fs}')

    return X

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
        * init_phase: phase info from the original audio file
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
def spectrogram_inversion_batch(config, index, spec, phase_info=True):
    '''
    https://dsp.stackexchange.com/questions/9877/reconstruction-of-audio-signal-from-spectrogram/13401#13401
    '''
    mag_spec = recover_mag_spec(spec)
    len = mag_spec.shape[1]

    if phase_info:
        X = retrieve_single_audio(config, index)
        phase = np.angle(librosa.stft(y=X, n_fft=2048, hop_length=512, center=False))
        phase = phase[:, :len]
        audio = librosa.istft(mag_spec * np.exp(1j * phase), n_fft=2048, hop_length=512)
    else:
        # recover a first audio reconstruction from the SPSI
        SPSI_audio = spsi(msgram=mag_spec, n_fft=2048, hop_length=512)

        # get the initial phase from the SPSI audio
        phase = np.angle(librosa.stft(y=SPSI_audio, n_fft=2048, hop_length=512, center=False))

        # get the audio using Griffin Lim
        audio = griffin_lim(magnitude_spectrogram=mag_spec,
                            n_fft=2048,
                            hop_length=512,
                            num_iterations=100,
                            init_phase=phase,
                            )

    return audio, phase

def spectrogram_inversion(config, index, spec, phase_info=True, phase_to_use=None):
    '''
    Inversion of a spectrogram.
    Could be done with original phase (just an ISTFT) or without (improved Griffin-Lim)

    :param config: config file (for getting the audio file path)
    :param index: index of the audio file being used
    :param spec: spectrogram on which to perform the inversion
    :param phase_info: flag (True or False)
    :param phase_to_use: phase info from an already converted audio
    :return: audio file (array)
    '''
    spec_shape = spec.shape[1]

    # recover the magnitude spectrogram from the power spectrogram
    mag_spec = recover_mag_spec(spec)

    if phase_info:
        if phase_to_use is not None:
            audio = librosa.istft(mag_spec * np.exp(1j*phase_to_use), n_fft=2048, hop_length=512)
            phase = None
        else:
            # get the phase info from the original audio file
            X = retrieve_single_audio(config, index)
            spectrogram = compute_spectrum(X)
            spec_shape_og = spectrogram.shape[1]

            # comment this block for normal usage
            batch = create_mini_batch_RawNet(X)
            X = batch.numpy().squeeze()
            #######################################

            phase = np.angle(librosa.stft(y=X, n_fft=2048, hop_length=512, center=False))
            phase = phase[:, :spec_shape]
            # reconstruct the audio using magnitude and phase
            audio = librosa.istft((mag_spec * np.exp(1j*phase))[:, :spec_shape_og], n_fft=2048, hop_length=512)

    else:
        # recover a first audio reconstruction from the SPSI
        SPSI_audio = spsi(msgram=mag_spec, n_fft=2048, hop_length=512)

        # get the initial phase from the SPSI audio
        phase = np.angle(librosa.stft(y=SPSI_audio, n_fft=2048, hop_length=512, center=False))

        # get the audio using Griffin Lim
        audio = griffin_lim(magnitude_spectrogram=mag_spec,
                            n_fft=2048,
                            hop_length=512,
                            num_iterations=100,
                            init_phase=phase,
                            )

    return audio, phase

