"""
SIGNAL PROCESSING UTILS
#TODO delete from tests.tests.py
"""
import librosa
import numpy as np
import math
import scipy.signal.windows as windows
from src.audio_utils import read_audio
from src.resnet_features import compute_spectrum

def get_spectrogram_from_audio(audio_path):
    audio, fs = read_audio(audio_path,
                           dur=180,
                           fs=16000,
                           trim=False,
                           int_type=False,
                           windowing=False)

    spec = compute_spectrum(audio)
    return spec



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


def spectrogram_inversion(spec):
    # recover the magnitude spectrogram from the power spectrogram
    mag_spec = recover_mag_spec(spec)

    # recover a first audio from the SPSI
    SPSI_audio = spsi(msgram=mag_spec, n_fft=2048, hop_length=512)

    # get the initial phase from the SPSI audio
    phase = np.angle(librosa.stft(y=SPSI_audio, n_fft=2048, hop_length=512, center=False))

    # get the audio using Griffin Lim
    audio = griffin_lim(magnitude_spectrogram=mag_spec,
                        n_fft=2048,
                        hop_length=512,
                        num_iterations=100,
                        init_phase=phase)

    return audio