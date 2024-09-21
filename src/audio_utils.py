import librosa
import numpy as np
import warnings
import soundfile as sf

warnings.filterwarnings("ignore")

def read_audio(audio_path, dur=180, fs=16000, trim=False, int_type=False, windowing=False):

    X, fs_orig = librosa.load(audio_path, sr=None, duration=dur)
    audio_len = len(X)
    X = X[:47104]

    if fs_orig != fs:
        X = librosa.resample(X, fs_orig, fs)

    if trim:
        X = librosa.effects.trim(X, top_db=20)[0]
    # from float to int
    if int_type:
        X = (X * 32768).astype(np.int32)
    if windowing:
        win_len = 3 # in seconds
        mask = np.zeros(dur*fs).astype(bool)
        for ii in range(mask.shape[0]//(win_len*fs)):
            mask[ii*win_len*fs:ii*win_len*fs+fs] = True
            mask = mask[:X.shape[0]]
        X = X[mask]

        sf.write(audio_path, X, fs)

    return X, fs, audio_len


def mix_tracks(audio1, audio2):
    mix_len = np.min([len(audio1), len(audio2)])
    mix = (audio1[:mix_len] + audio2[:mix_len]) / 2

    return mix