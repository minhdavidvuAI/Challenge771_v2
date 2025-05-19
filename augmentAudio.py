import os
import torch
from tqdm import tqdm
import config
from dataset.dataset_ESC50 import ESC50
import librosa
import random
import numpy as np
import soundfile as sf

# Path to raw audio files
audio_dir = r"data/esc50/ESC-50-master/audio"
augmented_data_dir = r"data/ESC-50-augmented-data"

def add_noise(data):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
        
    data = data.copy()  # <-- ADD THIS LINE to avoid modifying original array
    noise = np.random.normal(0, 0.005, len(data))
    audio_noisy = data + noise
    return torch.from_numpy(audio_noisy).float()

def pitch_shifting(data):
    sr = 22050
    bins_per_octave = 12
    pitch_pm = 10
    pitch_change = pitch_pm * 2 * np.random.uniform()

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
        
    data = data.copy()  # <-- ADD THIS LINE to avoid modifying original array
    data = librosa.effects.pitch_shift(y=data.astype('float64'), sr=sr, n_steps=pitch_change, bins_per_octave=bins_per_octave)
    return torch.from_numpy(data).float()

def random_shift(data):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
        
    data = data.copy()  # <-- ADD THIS LINE to avoid modifying original array
    timeshift_fac = 0.2 * 2 * (np.random.uniform() - 0.5)  # up to ±20% of length
    start = int(len(data) * timeshift_fac)
    if start > 0:
        data = np.pad(data, (start, 0), mode='constant')[:len(data)]
    else:
        data = np.pad(data, (0, -start), mode='constant')[:len(data)]
    return torch.from_numpy(data).float()

def time_masking(data, mask_param=16000):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    data = data.copy()  # <-- ADD THIS LINE to avoid modifying original array
    data_len = len(data)
    start = np.random.randint(0, data_len - mask_param)
    data[start:start+mask_param] = 0
    return torch.from_numpy(data).float()
    
def audio_augmentation(file, aug):
    directory = augmented_data_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    aug = np.array(aug, dtype='float32').reshape(-1, 1)
    print(sr)
    sf.write(os.path.join(directory, file), aug, sr, 'PCM_24')
    
if not os.path.exists(augmented_data_dir):
    os.makedirs(augmented_data_dir)

# List all files
files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])

# List of augmentation functions# List of all augmentation functions (adding the new ones)
#aug_functions = [add_noise, pitch_shifting, random_shift]
aug_functions = [add_noise, pitch_shifting, random_shift, time_masking]

for file_name in tqdm(files, desc="Augmenting Data"):
    file_path = os.path.join(audio_dir, file_name)
    waveform, sr = librosa.load(file_path)  # load with fixed sr

    # Apply random subset of augmentations
    num_augments = random.randint(1, len(aug_functions))
    augmentations_to_apply = random.sample(aug_functions, num_augments)

    augmented_waveform = waveform.copy()
    for aug_func in augmentations_to_apply:
        augmented_waveform = aug_func(augmented_waveform)
    
    # Save augmented waveform using your function
    audio_augmentation(file_name, augmented_waveform)