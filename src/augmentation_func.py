# src/audio_augmentation.py

import torch
import torch.nn.functional as F
import random

class RandomStartCrop:
    def __init__(self, sampling_rate: int, duration: int):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.maxlen = int(sampling_rate * duration)
        self.last_start = None  # Startindex (Samples)
        self.last_end = None    # Endindex (Samples)

    def __call__(self, audio_samples: torch.Tensor) -> torch.Tensor:
        if audio_samples.dim() > 1:
            audio_samples = audio_samples.squeeze()

        length = audio_samples.shape[0]

        # kürzer/gleich -> ggf. Padding, Start = 0, Ende = maxlen
        if length <= self.maxlen:
            if length < self.maxlen:
                pad_len = self.maxlen - length
                audio_samples = F.pad(audio_samples, (0, pad_len))
            self.last_start = 0
            self.last_end = self.maxlen
            return audio_samples

        # länger -> zufälliges Fenster
        start = torch.randint(0, length - self.maxlen + 1, (1,)).item()
        end = start + self.maxlen
        self.last_start = start
        self.last_end = end
        return audio_samples[start:end]


class RandomNoise:

    def __init__(self, min_factor: float = 0.0, max_factor: float = 0.05):
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.last_factor = None

    def __call__(self, audio_samples: torch.Tensor) -> torch.Tensor:
        if audio_samples.dim() > 1:
            audio_samples = audio_samples.squeeze()

        factor = random.uniform(self.min_factor, self.max_factor)
        self.last_factor = factor

        std = audio_samples.std()
        if std == 0:
            std = 1

        noise = torch.randn_like(audio_samples) * std * factor
        return audio_samples + noise


class FlipWave:

    def __init__(self):
        self.last_applied = None

    def __call__(self, audio_samples: torch.Tensor) -> torch.Tensor:
        if audio_samples.dim() > 1:
            audio_samples = audio_samples.squeeze()

        return -audio_samples