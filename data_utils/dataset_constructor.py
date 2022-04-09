"""
1. Select a subset of data from hub
2. Compress and take spectrogram
3. Save it back to hub
"""
import hub
import numpy as np
import pandas as pd
from data_utils import nsynth_adapter as na
from data_utils import preprocessing
import time
from PIL import Image


SELECTED_FAMILIES = [
    na.InstrumentFamily.KEYBOARD,
    na.InstrumentFamily.ORGAN,
    na.InstrumentFamily.GUITAR,
    na.InstrumentFamily.REED,
    na.InstrumentFamily.BRASS
]
INSTRUMENTS_PER_FAMILY = 8


class DatasetConstructor:
    default_source_train = 'hub://activeloop/nsynth-train'
    default_source_validate = 'hub://activeloop/nsynth-val'
    default_source_test = 'hub://activeloop/nsynth-test'
    default_target_train = 'hub://jakeval/nsynth-train'
    default_target_validate = 'hub://jakeval/nsynth-val'
    default_target_test = 'hub://jakeval/nsynth-test'

    def __init__(self,
                 preprocessor: preprocessing.SpectrogramPreprocessor,
                 source='train',
                 target='train',
                 token=None):
        """Construct a spectrogram dataset by taking a subset of an audio dataset and processing it.

        source: a string, usually a local file or a url
        target: a string, usually a local file or a url
        adapter: a dataset adapter (such as nsynth_adapter)
        """
        self.preprocessor = preprocessor
        if source == 'train':
            self.source = DatasetConstructor.default_source_train
        elif source == 'validate':
            self.source = DatasetConstructor.default_source_validate
        elif source == 'test':
            self.source = DatasetConstructor.default_source_test
        else:
            self.source = source
        if target == 'train':
            self.target = DatasetConstructor.default_target_train
        elif target == 'validate':
            self.target = DatasetConstructor.default_target_validate
        elif target == 'test':
            self.target = DatasetConstructor.default_target_test
        else:
            self.target = target
        self.df = None
        self.ds = None
        self.sample_rate = None
        self.samples_per_clip = None
        self.token = token

    def initialize_dataset(self):
        self.ds = hub.load(self.source)
        self.sample_rate = self._clean_data(self.ds.sample_rate[0])
        self.preprocessor.set_sample_rate(self.sample_rate)
        self.samples_per_clip = self.ds.audios.shape[1]
        self.df = pd.DataFrame({
            'id': range(self.ds.instrument.shape[0]),
            'family': self._clean_data(self.ds.instrument_family),
            'instrument': self._clean_data(self.ds.instrument),
            'pitch': self._clean_data(self.ds.pitch)
        })
        return self.df

    def _clean_data(self, data, dtype=np.int64):
        val = np.squeeze(data.numpy().astype(dtype))
        if len(val.shape) == 0:
            return dtype(val)
        else:
            return val

    def select_subset_from_target_subset(self, target_subset=default_target_test):
        """Select a subset from the source dataset based on an existing target subset
        """
        # select given families
        selected_instruments = np.unique(self._clean_data(hub.load(target_subset, token=self.token).instrument))
        self.df = self.df.loc[self.df.instrument.isin(selected_instruments), :]
        return self.df

    def select_random_subset(self, selected_families=SELECTED_FAMILIES, instruments_per_family=INSTRUMENTS_PER_FAMILY, in_place=True):
        # select given families
        df = self.df.loc[self.df.family.isin(selected_families), :].copy()
        
        # select instruments per family
        selected_instruments = {}
        for family in selected_families:
            instruments = df.loc[df.family == family, 'instrument'].unique()
            if instruments_per_family is None or instruments_per_family > instruments.shape[0]:
                selected_instruments[family] = instruments
            else:
                selected_instruments[family] = np.random.choice(instruments, instruments_per_family, replace=False)
        df = df.groupby('family', as_index=False).apply(lambda subdf: subdf[subdf['instrument'].isin(selected_instruments[subdf.name])])
        if df.index.nlevels > 1:
            df.index = df.index.droplevel()
        if in_place:
            self.df = df
        return df

    def write_subset_to_dataset(self, streaming_chunk_size=0.25, verbose=True):
        number_of_clips = self.df.shape[0]
        clip_size = self.samples_per_clip * 64 / 8e9 # gigabytes per clip
        clips_per_chunk = int(streaming_chunk_size / clip_size)
        number_of_chunks = int(np.ceil(number_of_clips / clips_per_chunk))
        indices = list(np.sort(self.df['id'].to_numpy()))
        total_time = 0

        if verbose:
            print(f"{number_of_clips} clips will be written in {number_of_chunks} chunks.")
            print("Write the metadata...")
        audio = self._clean_data(self.ds.audios[0], dtype=np.float32)
        f, t, _ = self.preprocessor.get_spectrograms(audio)
        dmeta = hub.empty(f"{self.target}-metadata", overwrite=True, token=self.token)
        with dmeta:
            dmeta.create_tensor('f', htype='generic')
            dmeta.create_tensor('t', htype='generic')
            dmeta.append({'f': f, 't': t})
        if verbose:
            print("Finished. Writing spectrograms...")
        dt = hub.empty(self.target, overwrite=True, token=self.token)
        with dt:
            dt.create_tensor('spectrogram', htype='generic')
            dt.create_tensor('instrument', htype='class_label')
            dt.create_tensor('instrument_family', htype='class_label')
            dt.create_tensor('pitch', htype='class_label')
            dt.create_tensor('id', htype='class_label')
            for i in range(number_of_chunks):
                start_time = time.time()
                indices_to_load = indices[i * clips_per_chunk : (i + 1) * clips_per_chunk]
                print(f"Load {len(indices_to_load)} audio clips...")
                audio = self._clean_data(self.ds.audios[indices_to_load], dtype=np.float32)
                print("Take the spectrogram...")
                _, _, S = self.preprocessor.get_spectrograms(audio)
                print("Write to the database...")
                count = 0
                for si, idx in zip(range(S.shape[0]), indices_to_load):
                    count += 1
                    df = self.df[self.df['id'] == idx].iloc[0]
                    dt.append({
                        'spectrogram': S[si],
                        'instrument': df['instrument'],
                        'instrument_family': df['family'],
                        'pitch': df['pitch'],
                        'id': idx })
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                if verbose:
                    remaining_time = (elapsed_time / (i+1)) * (number_of_chunks - (i+1))
                    print(f"Wrote data chunk {i+1}/{number_of_chunks} in {elapsed_time} seconds. ~{remaining_time/60} minutes remaining.")
        print(f"Finished writing data in {elapsed_time/60} minutes")

    def calculate_new_dataset_size(self):
        audio = self._clean_data(self.ds.audios[0], dtype=np.float32)
        f, t, s = self.preprocessor.get_spectrograms(audio)
        number_of_spectrograms = self.df.shape[0]
        spectrogram_size = s.size * s.itemsize / 1e9
        shape = (number_of_spectrograms, s.shape[0], s.shape[1])
        return shape, number_of_spectrograms * spectrogram_size

    def visualize_new_dataset(self, instruments_per_family=2):
        """Return a visualizable dataframe representing what will be written to the new dataset.

        instruments_per_family: how many instruments per family to visualize.
        """
        df = self.select_random_subset(selected_families=self.df['family'].unique(), instruments_per_family=instruments_per_family, in_place=False)
        df = df[(df['pitch'] > 40) & (df['pitch'] < 60)]
        df = df.drop_duplicates(subset='instrument')
        idx_to_load = list(df['id'].to_numpy())
        print(f"start loading {len(idx_to_load)} samples")
        audio = self._clean_data(self.ds.audios[idx_to_load], dtype=np.float32)
        print("finished loading!")
        print(audio.shape)
        f, t, S = self.preprocessor.get_spectrograms(audio)
        df['audio'] = [na.PlayableAudio(f, t, S[i], audio[i], self.sample_rate) for i in range(audio.shape[0])]
        return df

def write_image():
    X = np.random.random((64, 64))
    im = Image.fromarray(X, mode='RGB')
    im.save('./data/test/test_im.png')

if __name__ == '__main__':
    preprocessor = preprocessing.SpectrogramPreprocessor(max_freq=8000, window_size=1024, n_mels=32)
    dc = DatasetConstructor(preprocessor, 'train')
    print("initialize...")
    start_df = dc.initialize_dataset()
    print("select subset")
    subset = dc.select_random_subset([na.InstrumentFamily.KEYBOARD, na.InstrumentFamily.BRASS], instruments_per_family=INSTRUMENTS_PER_FAMILY)
    shape, size = dc.calculate_new_dataset_size()
    print(f"New dataset will be {size} gb and have shape {shape}.")
