import enum
from IPython import display
from matplotlib import pyplot as plt
from data_utils import preprocessing

class InstrumentFamily(enum.IntEnum):
    BASS = 0
    BRASS = 1
    FLUTE = 2
    GUITAR = 3
    KEYBOARD = 4
    MALLET = 5
    ORGAN = 6
    REED = 7
    STRING = 8
    SYNTH_LEAD = 9
    VOCAL = 10


class PlayableAudio:
    def __init__(self, f, t, s, audio, sample_rate):
        self.f = f
        self.t = t
        self.s = s
        self.audio = audio
        self.sr = sample_rate
        
    def visualize(self, max_freq=None, max_time=None):
        fig, ax = plt.subplots()
        f, t, s = preprocessing.crop_image(self.f, self.t, self.s, max_freq, max_time)
        ax.pcolormesh(t, f, s, shading='gouraud')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (sec)')
        return fig, ax
    
    def play(self):
        show_audio = display.Audio(self.audio, rate=self.sr)
        display.DisplayHandle().display(show_audio)
