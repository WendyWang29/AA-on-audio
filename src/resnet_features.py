from scipy.interpolate import interpn
from scipy.signal import lfilter
from scipy.fft import dct
import math
import os
import logging
logging.basicConfig(level=logging.DEBUG)
import sys

from src.utils import *
from src.audio_utils import *


def window_stack(a, stepsize=int(0.5*16000), width=int(1*16000)):
    n = a.shape[0]
    return np.vstack(a[i:1+n+i-width:stepsize] for i in range(0, width))



def compute_spectrum(x, type_of_spec):
    if type_of_spec == 'logmag':
        s = librosa.stft(x, n_fft=2048, win_length=2048, hop_length=512, window='hann', center=True)
        phase = np.angle(s)
        a = np.abs(s)
        spec = librosa.amplitude_to_db(a, ref=np.max)
    elif type_of_spec == 'pow':
        s = librosa.stft(x, n_fft=2048, win_length=2048, hop_length=512, window='hann', center=True)
        a = np.abs(s) ** 2
        spec = librosa.power_to_db(a, ref=np.max)
        phase = np.angle(s)
    else:
        sys.exit(f'{type_of_spec} is a wrong type of spectrogram')

    return spec, phase


def get_log_spectrum(type_of_spec, X, fs, win_len=None, hop_size=None):
    if win_len:
        X = window_stack(X, stepsize=int(hop_size*fs), width=int(win_len*fs)).T

        feat_list = []
        for x_win in X:
            feat = compute_spectrum(x_win, type_of_spec)
            feat_list.append(feat.T)
        return np.array(feat_list)
    else:
        spec, phase = compute_spectrum(X, type_of_spec)
    return spec, phase


def compute_mfcc(x):
    if len(x) < 16000:
        x = np.concatenate([x, x], axis=0)
    mfcc = librosa.feature.mfcc(x, sr=16000, n_mfcc=24, center=False)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(delta)
    feats = np.concatenate((mfcc, delta, delta2), axis=0)

    return feats


def compute_mfcc_feats(wav_path, X, fs, win_len=None, hop_size=None):
    # X, fs = read_audio(wav_path, int_type=False, trim=False)
    if win_len:
        X = window_stack(X, stepsize=int(hop_size*fs), width=int(win_len*fs)).T

        feat_list = []
        for x_win in X:
            feat = compute_mfcc(x_win)
            feat_list.append(feat.T)
        return np.array(feat_list)
    else:
        feats = compute_mfcc(X)
        return feats



def extract_features(wav_path, feature, args, force=False):
    """
    Extract features chosen by features argument.

    :param wav_path: filename of the audio
    :type wav_path: str
    :param feature: name of the features to be computed in [spec, mfcc]
    :type feature: str
    :param args: configuration dictionary
    :type args: dict
    :return: extracted features
    :rtype np.array
    """
    def get_feats(wav_path, X):

        args['win_len'] = None
        args['hop_size'] = None

        if feature == 'spec':
            return get_log_spectrum(wav_path, X, args['fsamp'], win_len=args['win_len'], hop_size=args['hop_size'])
        elif feature == 'mfcc':
            return compute_mfcc_feats(wav_path, X, args['fsamp'], win_len=args['win_len'], hop_size=args['hop_size'])
        else:
            raise ValueError('Feature type not supported.')

    cache_dir = args['cache_dir'] + feature
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    file_name = os.path.join(cache_dir, os.path.splitext(os.path.basename(wav_path))[0] + '.npy')
    if not os.path.exists(file_name) or force:
        X, fs = read_audio(wav_path)
        data = get_feats(wav_path, X)
        if np.isnan(data).any():
            return
        else:
            np.save(file_name, data)
    else:
        # data = np.load(file_name, allow_pickle=True)
        pass
    # return data