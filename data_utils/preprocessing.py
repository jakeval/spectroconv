from scipy import signal
import numpy as np
import librosa


class SpectrogramPreprocessor:
    def __init__(self, max_freq=None, max_time=None, window_size=1024, n_mels=None, scaling=1000):
        self.max_freq = max_freq
        self.max_time = max_time
        self.window_size = window_size
        self.n_mels = n_mels
        self.sample_rate = None
        self.scaling = 1000

    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate

    def get_spectrograms(self, audio):
        f, t, s = signal.spectrogram(audio,
                                     fs=self.sample_rate,
                                     nperseg=self.window_size,
                                     mode='magnitude',
                                     scaling='spectrum')
        f, t, s = crop_image(f, t, s, self.max_freq, self.max_time)
        if self.n_mels is not None:
            f, s = self.mel_compression(f, s, self.n_mels)
        return f, t, s * self.scaling

    def mel_compression(self, f, s, n_mels):
        n_fft = (s.shape[-2]-1)*2
        melfb = librosa.filters.mel(sr=self.sample_rate, n_fft=n_fft, n_mels=n_mels, fmax=f[-1])
        s = melfb @ s
        f = librosa.mel_frequencies(n_mels, fmax=f[-1])
        return f, s


def crop_image(f, t, s, max_freq=None, max_time=None):
    # crop the image vertically
    if max_freq is not None:
        f_axis = len(s.shape) - 2
        f = f[f < max_freq]
        s = s.take(indices=range(f.shape[0]), axis=f_axis)
    # crop the image horizontally
    if max_time is not None:
        t_axis = len(s.shape) - 1
        t = t[t < max_time]
        s = s.take(indices=range(t.shape[0]), axis=t_axis)
    return f, t, s
