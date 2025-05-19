import os
import torch
from tqdm import tqdm
import librosa
import random
import numpy as np
import soundfile as sf

class AudioAugmenter:
    def __init__(self, audio_dir, augmented_data_dir, sr=22050, noise_level=0.005, pitch_pm=10, mask_param=16000):
        """
        Initializes the AudioAugmenter with the directory paths and augmentation parameters.
        """
        self.audio_dir = audio_dir
        self.augmented_data_dir = augmented_data_dir
        self.sr = sr
        self.noise_level = noise_level
        self.pitch_pm = pitch_pm
        self.mask_param = mask_param
        
        # Create augmented data directory if it doesn't exist
        if not os.path.exists(self.augmented_data_dir):
            os.makedirs(self.augmented_data_dir)

        # List all files in the audio directory
        self.files = sorted([f for f in os.listdir(self.audio_dir) if f.endswith('.wav')])
        
        print(f"found files {self.files}")
        # List of augmentation functions
        self.aug_functions = [self.add_noise, self.pitch_shifting, self.random_shift, self.time_masking]

    def add_noise(self, data):
        """
        Adds random Gaussian noise to the audio data.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        data = data.copy()  # Avoid modifying the original array
        noise = np.random.normal(0, self.noise_level, len(data))
        audio_noisy = data + noise
        return torch.from_numpy(audio_noisy).float()

    def pitch_shifting(self, data):
        """
        Shifts the pitch of the audio data.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        data = data.copy()  # Avoid modifying the original array
        pitch_change = self.pitch_pm * 2 * np.random.uniform()
        data = librosa.effects.pitch_shift(y=data.astype('float64'), sr=self.sr, n_steps=pitch_change)
        return torch.from_numpy(data).float()

    def random_shift(self, data):
        """
        Shifts the audio data randomly.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        data = data.copy()  # Avoid modifying the original array
        timeshift_fac = 0.2 * 2 * (np.random.uniform() - 0.5)  # up to ±20% of length
        start = int(len(data) * timeshift_fac)
        if start > 0:
            data = np.pad(data, (start, 0), mode='constant')[:len(data)]
        else:
            data = np.pad(data, (0, -start), mode='constant')[:len(data)]
        return torch.from_numpy(data).float()

    def time_masking(self, data):
        """
        Applies time masking to the audio data.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        data = data.copy()  # Avoid modifying the original array
        data_len = len(data)
        start = np.random.randint(0, data_len - self.mask_param)
        data[start:start + self.mask_param] = 0
        return torch.from_numpy(data).float()

    def audio_augmentation(self, file, aug):
        """
        Saves the augmented audio data to the specified directory.
        """
        aug = np.array(aug, dtype='float32').reshape(-1, 1)
        sf.write(os.path.join(self.augmented_data_dir, file), aug, self.sr, 'PCM_24')

    def augment_data(self):
        """
        Augments all audio files in the directory and saves them.
        """
        for file_name in tqdm(self.files, desc="Augmenting Data"):
            file_path = os.path.join(self.audio_dir, file_name)
            waveform, _ = librosa.load(file_path, sr=self.sr)  # Load with fixed sample rate

            # Apply a random subset of augmentations
            num_augments = random.randint(1, len(self.aug_functions))
            augmentations_to_apply = random.sample(self.aug_functions, num_augments)

            augmented_waveform = waveform.copy()
            for aug_func in augmentations_to_apply:
                augmented_waveform = aug_func(augmented_waveform)

            # Save the augmented waveform using the method
            self.audio_augmentation(file_name, augmented_waveform)