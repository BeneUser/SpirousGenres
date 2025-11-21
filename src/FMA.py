import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io.wavfile as wav
import pandas as pd
import src.Utils as Utils 

class FMAAudioDataset(Dataset):

    def __init__(self, audio_files, labels, audio_path, sampling_rate, duration):
        self.audio_files = audio_files
        self.labels = labels
        self.audio_path = audio_path
        self.maxlen = sampling_rate * duration
        self.sampling_rate = sampling_rate
        self.duration = duration

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        fname = self.audio_files[idx]
        label = self.labels[idx]

        # FMA folder structure: first 3 digits
        subfolder = fname[:3]
        path = self.audio_files[idx]

        # Load WAV using scipy
        rate, audio = wav.read(path)

        # Convert from int16 → float32
        audio = audio.astype(np.float32)

        # Convert stereo → mono (average)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample if sample rates don't match
        if rate != self.sampling_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=rate, target_sr=self.sampling_rate)

        # Truncate or pad
        if len(audio) > self.maxlen:
            audio = audio[:self.maxlen]
        else:
            audio = np.pad(audio, (0, self.maxlen - len(audio)), mode='constant')

        audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        return audio, label
    


def load_fma_small_testset(config, corrupted_files=None):

    label_map={'blues' : 0, 'classical' : 1, 'country' : 2,
           'disco' : 3, 'hiphop'    : 4, 'jazz'    : 5,
           'metal' : 6, 'pop'       : 7, 'reggae'  : 8, 'rock' : 9}
    
    TARGET_GENRES = {"hiphop", "rock", "pop"} 

    # from https://nbviewer.org/github/mdeff/fma/blob/outputs/usage.ipynb
    tracks = Utils.load(os.path.join(config.fma_audio_dir_path, 'fma_metadata', 'tracks.csv'))
    #genres = Utils.load(os.path.join(config.fma_audio_dir_path, 'fma_metadata', 'genres.csv'))
    #features = Utils.load(os.path.join(config.fma_audio_dir_path, 'fma_metadata', 'features.csv'))
    #echonest = Utils.load(os.path.join(config.fma_audio_dir_path, 'fma_metadata', 'echonest.csv'))

    #np.testing.assert_array_equal(features.index, tracks.index)
    #assert echonest.index.isin(tracks.index).all()

    print(tracks.shape) #, genres.shape, features.shape, echonest.shape)

    # Select small dataset and test split
    small = tracks[tracks['set', 'subset'] <= 'small']
    small_test = small[small['set', 'split'] == 'test'].copy()

    # Normalize genre names
    small_test['track', 'genre_top'] = (
        small_test['track', 'genre_top']
        .astype(str)
        .str.lower()
        .str.replace('-', '', regex=False)
        .str.strip()
    )

    # Filter by target genres
    small_test = small_test[small_test[('track', 'genre_top')].isin(TARGET_GENRES)]
    print("After genre filter:", small_test.shape)

    # Build file paths
    track_ids = small_test.index.values
    file_paths = [
        os.path.join(config.fma_audio_dir_path, f"{tid:06d}"[:3], f"{tid:06d}.wav")
        for tid in track_ids
    ]

    # Map labels
    genre_strings = small_test['track', 'genre_top'].values
    labels = np.array([label_map[g] for g in genre_strings])

    # Remove corrupted files
    if corrupted_files is not None:
        corrupted_files_set = set(corrupted_files)
        keep_mask = [fname not in corrupted_files_set for fname in file_paths]
        file_paths = np.array(file_paths)[keep_mask].tolist()
        labels = labels[keep_mask]

    print("Loaded FMA-small TEST subset:")
    print(f"  Total usable files: {len(file_paths)}")
    print(f"  Total usable labels: {len(labels)}")
    print(f"  Genres included: {TARGET_GENRES}")

    return file_paths, labels
