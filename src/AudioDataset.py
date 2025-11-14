from torch.utils.data import Dataset
import torch
import os
import scipy.io.wavfile as wav

class AudioDataset(Dataset):
    def __init__(self, audio_files, labels, audio_path,
                 maxlen, sampling_rate, duration):
        self.audio_files = audio_files
        self.audio_path = audio_path
        self.labels = labels
        self.maxlen = maxlen
        self.sampling_rate = sampling_rate
        self.duration = duration

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        audio_file = self.audio_files[idx]
        audio_dir = audio_file[:audio_file.index('.')]
        file_path = os.path.join(self.audio_path, audio_dir, audio_file)
        (rate,audio_samples) = wav.read(file_path)
        audio_samples = torch.from_numpy(audio_samples).to(torch.float32)
        if len(audio_samples) > self.maxlen:
            # Truncate
            audio_samples = audio_samples[:self.maxlen]

        tstart = 0 # Offset from start of song (hyper-parameter!)
        audio_samples = audio_samples[int(self.sampling_rate*tstart):int(self.sampling_rate*(tstart+self.duration))]



        # write one function for each type of augmention - test outside of the training 
        # flip 
        # 3-4types 

        return audio_samples, label