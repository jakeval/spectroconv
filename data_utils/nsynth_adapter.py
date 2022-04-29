import enum
from IPython import display
from matplotlib import pyplot as plt
from data_utils import preprocessing
import hub
import numpy as np
import pandas as pd


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
        plt.close(fig)
        return fig, ax
    
    def play(self):
        show_audio = display.Audio(self.audio, rate=self.sr)
        display.DisplayHandle().display(show_audio)


class NsynthDataset:
    default_source_train = 'hub://jakeval/nsynth-train'
    default_source_validate = 'hub://jakeval/nsynth-val'
    default_source_test = 'hub://jakeval/nsynth-test'
    default_audio_train = 'hub://activeloop/nsynth-train'
    default_audio_validate = 'hub://activeloop/nsynth-val'
    default_audio_test = 'hub://activeloop/nsynth-test'

    def __init__(self, source='train', token_file='./.activeloop.key'):
        if source == 'train':
            self.source = NsynthDataset.default_source_train
        elif source == 'val':
            self.source = NsynthDataset.default_source_validate
        elif source == 'test':
            self.source = NsynthDataset.default_source_test
        else:
            self.source = source
        self.df = None
        self.ds = None
        self.f = None
        self.t = None
        self.X = None
        self.Y = None
        self.ids = None
        self.token = None
        if token_file is not None:
            with open(token_file) as f:
                self.token = f.read().strip()

    def initialize(self, code_lookup=None):
        metads = None
        if self.token is not None:
            metads = hub.load(f"{self.source}-metadata", token=self.token, read_only=True)
            self.ds = hub.load(self.source, token=self.token, read_only=True)
        else:
            metads = hub.load(f"{self.source}-metadata", read_only=True)
            self.ds = hub.load(self.source, read_only=True)
        
        self.f = self._clean_data(metads.f, dtype=np.float32)
        self.t = self._clean_data(metads.t, dtype=np.float32)
        
        self.df = pd.DataFrame({
            'id': self._clean_data(self.ds.id),
            'family': self._clean_data(self.ds.instrument_family),
            'instrument': self._clean_data(self.ds.instrument),
            'pitch': self._clean_data(self.ds.pitch)
        })

        self.codes = self.df['family'].unique()
        if code_lookup is None:
            self.code_lookup = dict([(code, i) for i, code in enumerate(self.codes)])
        else:
            self.code_lookup = code_lookup
        return self.code_lookup

    def set_code_lookup(self, code_lookup):
        self.code_lookup = code_lookup

    def get_dataloader(self, batch_size, shuffle=True, include_ids=False):
        def transform_spectrogram(X):
            return X.reshape((1, X.shape[0], X.shape[1]))
        
        def transform_family(y):
            return np.int64(self.id_to_ordinal(y.squeeze())) #self.id_to_ordinal(y)

        transform = {
            'spectrogram': transform_spectrogram,
            'instrument_family': transform_family,
        }
        if include_ids:
            transform.update({'id': None})

        return self.ds.pytorch(
            batch_size = batch_size,
            transform = transform,
            shuffle = shuffle,
            use_local_cache = True
        )

    def get_dataframe(self, selected_ids, audio_source):
        if audio_source == 'train':
            audio_source = NsynthDataset.default_audio_train
        elif audio_source == 'val':
            audio_source = NsynthDataset.default_audio_validate
        elif audio_source == 'test':
            audio_source = NsynthDataset.default_audio_test
        else:
            audio_source = audio_source

        selected_ids = list(selected_ids)
        df = self.df.copy()
        mask = df['id'].isin(selected_ids)
        idxs = np.arange(self.df.shape[0])[mask]
        df = df.loc[mask,:]
        df_ids = list(df['id'].to_numpy())

        print(f"Loading {len(df_ids)} spectrograms...")
        spectrogram = self._clean_data(self.ds.spectrogram[list(idxs)], dtype=np.float32)
        df['spectrogram'] = list(spectrogram)

        print(f"Loading {len(df_ids)} audio clips...")
        audio_ds = hub.load(audio_source)
        sample_rate = self._clean_data(audio_ds.sample_rate[0])
        df['sample_rate'] = sample_rate
        audio = self._clean_data(audio_ds.audios[list(df_ids)], dtype=np.float32)
        df['audio'] = list(audio)
        print("Linished loading")

        df = df.set_index('id', drop=False).loc[selected_ids]

        return df

    def get_data(self, selected_families=None, instruments_per_family=None, selected_ids=None, max_pitch=72, min_pitch=48):
        idxs = np.arange(self.df.shape[0])
        if selected_ids is not None:
            idxs = idxs[self.df['id'].isin(selected_ids)]
        elif selected_families is not None:
            df = self.df[self.df.family.isin(selected_families)].copy()
            df = df[(df['pitch'] >= min_pitch) & (df['pitch'] <= max_pitch)]
            selected_instruments = {}
            for family in selected_families:
                instruments = df.loc[df.family == family, 'instrument'].unique()
                if instruments_per_family is None or instruments_per_family > instruments.shape[0]:
                    selected_instruments[family] = instruments
                else:
                    selected_instruments[family] = np.random.choice(instruments, instruments_per_family, replace=False)
            df = df.groupby('family', as_index=False).apply(lambda subdf: subdf[subdf['instrument'].isin(selected_instruments[subdf.name])])
            idxs = idxs[self.df.id.isin(df.id)]
        idxs = idxs
        print(f"Begin loading {idxs.shape[0]} spectrograms...")
        X = self._clean_data(self.ds.spectrogram[list(idxs)], dtype=np.float32)
        y = self.df.iloc[list(idxs)].family.to_numpy()
        y = self.id_to_ordinal(y)
        ids = self.df.iloc[list(idxs)].id.to_numpy()
        return X, y, ids

    def sample_shape(self):
        return self.ds.spectrogram[0].shape

    def visualize_new_dataset(self, selected_ids, audio_source='train'):
        """Return a visualizable dataframe representing the data."""
        if audio_source == 'train':
            audio_source = NsynthDataset.default_audio_train
        elif audio_source == 'validate':
            audio_source = NsynthDataset.default_audio_validate
        elif audio_source == 'test':
            audio_source = NsynthDataset.default_audio_test
        else:
            audio_source = audio_source
        
        df = self.df[self.df['id'].isin(selected_ids)].drop_duplicates(subset='instrument').copy()
        spectrogram_idxs = list(np.arange(self.df.shape[0])[self.df['id'].isin(df['id'])])
        audio_idxs = list(df['id'].to_numpy())
        print(f"start loading {len(audio_idxs)} samples")
        audio_ds = hub.load(audio_source)
        sample_rate = self._clean_data(audio_ds.sample_rate[0])
        audio = self._clean_data(audio_ds.audios[audio_idxs], dtype=np.float32)
        print(audio.shape)
        print("finished loading!")
        print(f"start loading {len(spectrogram_idxs)} spectrograms")
        S = self._clean_data(self.ds.spectrogram[spectrogram_idxs], dtype=np.float32)
        print(S.shape)
        print("finished loading!")
        df['audio'] = [PlayableAudio(self.f, self.t, S[i], audio[i], sample_rate) for i in range(audio.shape[0])]
        return df

    def id_to_ordinal(self, y):
        y_encoded = y.copy()
        for code in self.codes:
            y_encoded[y == code] = self.code_lookup[code]
        return y_encoded

    def plot_spectrogram(self, s):
        fig, ax = plt.subplots()
        ax.pcolormesh(self.t, self.f, s, shading='gouraud')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (sec)')
        plt.close(fig)
        return fig

    def _clean_data(self, data, dtype=np.int64):
        val = np.squeeze(data.numpy().astype(dtype))
        if len(val.shape) == 0:
            return dtype(val)
        else:
            return val


def codes_to_enums(code_lookup):
    class_enums = [None] * len(code_lookup)
    for code, index in code_lookup.items():
        class_enums[index] = InstrumentFamily(code)
    return class_enums