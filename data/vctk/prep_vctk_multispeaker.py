"""
Create an HDF5 file of patches for training super-resolution model.
"""

import os, argparse
import numpy as np
import h5py
import pickle

import librosa
import random
from scipy import interpolate
from scipy.signal import decimate
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('--file-list',
  help='list of input wav files to process')
parser.add_argument('--in-dir', default='~/',
  help='folder where input files are located')
parser.add_argument('--out',
  help='path to output h5 archive')
parser.add_argument('--scale', type=int, default=2,
  help='scaling factor')
parser.add_argument('--dimension', type=int,
  help='dimension of patches--use -1 for no patching')
parser.add_argument('--stride', type=int, default=3200,
  help='stride when extracting patches')
parser.add_argument('--interpolate', action='store_true',
  help='interpolate low-res patches with cubic splines')
parser.add_argument('--low-pass', action='store_true',
  help='apply low-pass filter when generating low-res patches')
parser.add_argument('--batch-size', type=int, default=128,
  help='we produce # of patches that is a multiple of batch size')
parser.add_argument('--sr', type=int, default=16000, help='audio sampling rate')
parser.add_argument('--sam', type=float, default=1.,
  help='subsampling factor for the data')
parser.add_argument('--full_sample', type=bool, default=True)
parser.add_argument('--noise_dir', type=str, default=3200,
  help='noise dir')

parser.add_argument('--snr', type=int, default=3200,
  help='snr dataset')

args = parser.parse_args()

from scipy.signal import butter, lfilter
import re

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def add_data(h5_file, inputfiles, args, save_examples=False):

    # Make a list of all files to be processed
    file_list = []
    ID_list = []
    file_extensions = set(['.wav'])

    with open(inputfiles) as f:
        for line in f:
            filename = line.strip()
            ext = os.path.splitext(filename)[1]
            if ext in file_extensions:
                file_list.append(os.path.join(args.in_dir, filename))

    num_files = len(file_list)
    noise_files = []

    for root, dirs, files in os.walk(args.noise_dir):
        for file in files:
            if file.endswith("wav"):
                print(file)
                noise_files.append(os.path.join(root, file))

    #noise_files = [
    #    os.path.join(args.noise_dir, file) for file in os.listdir(args.noise_dir) if file.endswith('.wav')
    #]

    d, d_lr = args.dimension, args.dimension
    s, s_lr = args.stride, args.stride

    total_patches = 500000

    # Create datasets with chunking and maxshape to allow resizing
    data_set = h5_file.create_dataset(
        'data', (0, d_lr, 1), maxshape=(None, d_lr, 1),
        dtype=np.float32, chunks=True
    )

    label_set = h5_file.create_dataset(
        'label', (0, d, 1), maxshape=(None, d, 1),
        dtype=np.float32, chunks=True
    )

    global_idx = 0

    for j, file_path in enumerate(file_list):
        if j % 10 == 0:
            print('%d/%d' % (j, num_files))

        directory_id_matches = re.search(fr'p(\d{{3}})\{os.path.sep}', file_path)
        ID = int(directory_id_matches.group(1))

        # Load audio file
        x, fs = librosa.load(file_path, sr=args.sr)

        # Crop so that it works with scaling ratio
        x_len = len(x)
        x = x[:x_len - (x_len % args.scale)]

        # Generate low-resolution version by adding noise

        noise_file = random.choice(noise_files)
        noise, noise_sr = librosa.load(noise_file,  sr=args.sr)

        if len(noise) < len(x):
            noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
        noise = noise[:len(x)]

        # Scale noise to achieve desired SNR
        snr = args.snr  # Desired Signal-to-Noise Ratio in dB
        signal_power = np.mean(x ** 2)
        noise_power = np.mean(noise ** 2)
        scaling_factor = np.sqrt(signal_power / (10 ** (float(args.snr) / 10) * noise_power))
        noise = noise * scaling_factor

        x_lr = x + noise  # Add noise to the original signal

        max_i = len(x) - d + 1

        for i in range(0, max_i, s):
            u = np.random.uniform()
            if u > args.sam:
                continue
            i_lr = i

            hr_patch = np.array(x[i: i + d])
            lr_patch = np.array(x_lr[i_lr: i_lr + d_lr])

            assert len(hr_patch) == d
            assert len(lr_patch) == d_lr

            data_set.resize(global_idx + 1, axis=0)
            label_set.resize(global_idx + 1, axis=0)

            data_set[global_idx] = lr_patch.reshape((d_lr, 1))
            label_set[global_idx] = hr_patch.reshape((d, 1))

            global_idx += 1

    # Trim datasets to the actual number of patches added (if sampled randomly)
    data_set.resize(global_idx, axis=0)
    label_set.resize(global_idx, axis=0)


if __name__ == '__main__':
  # create train
  with h5py.File(args.out, 'w') as f:
    add_data(f, args.file_list, args, save_examples=False)
