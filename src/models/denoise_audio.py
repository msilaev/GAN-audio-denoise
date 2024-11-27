import os
import numpy as np
import h5py
import torch
import librosa
import soundfile as sf
from scipy import interpolate

import scipy.signal as signal
from scipy.signal import decimate
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt

from calculate_snr_lsd import get_lsd, get_snr

#from torchmetrics.audio import SignalImprovementMeanOpinionScore


import argparse
import concurrent.futures
import glob
import os

import librosa
import numpy as np
import numpy.polynomial.polynomial as poly
import onnxruntime as ort
import pandas as pd
import soundfile as sf
from requests import session
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

# ----------------------------------------------------------------------------
def get_scores(model, wav, name, model_path ,args):
    #else:
    if torch.cuda.is_available():
            print("CUDA!")
            device = torch.device('cuda')
    else:
            device = torch.device('cpu')

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    model.to(device)

    # Load signal
    x_hr, fs = librosa.load(wav, sr=args.sr)

    if fs == 16000:
        n_fft = 2048
    elif fs == 48000:
        n_fft = 3*2048
    else:
        n_fft = 2048

    pad_length = args.patch_size - (x_hr.shape[0] % args.patch_size)
    x_hr = np.pad(x_hr, (0, pad_length), 'constant', constant_values=(0, 0))

    # Downscale signal
    x_lr = decimate(x_hr, args.r)
    x_lr_upsampled = upsample(x_lr, args.r)

    n_patches = x_lr_upsampled.shape[0]//args.patch_size

    x_lr_librosa = librosa.resample(x_lr, orig_sr=fs // args.r, target_sr=fs)

    P = []
    Y = []
    X = []

    with torch.no_grad():
        for i in range(0, n_patches, 1):

            lr_patch = np.array(x_lr_upsampled[i * args.patch_size : (i+1)* args.patch_size ])

            hr_patch = np.array(x_hr[i * args.patch_size: (i+1) * args.patch_size ])

            x_lr_tensor_part = torch.tensor(lr_patch.flatten(),
                                        dtype=torch.float32).unsqueeze(0).unsqueeze(2)

            P.append( model(x_lr_tensor_part.to(device)).squeeze().cpu().numpy())
            Y.append(hr_patch)
            X.append(lr_patch)

    P = np.concatenate(P)
    Y = np.concatenate(Y)
    X = np.concatenate(X)

    snr_val = get_snr(P, Y)
    snr_val_librosa = get_snr(X, Y)

    lsd_val = get_lsd(P, Y, n_fft=n_fft)
    lsd_val_librosa = get_lsd(X, Y, n_fft=n_fft)

    print("snr pr", snr_val)
    print("snr librosa", snr_val_librosa)

    score_dict = dict()

    score_dict['filename'] = name
    score_dict['snr_pr'] = snr_val
    score_dict ['snr_lbr'] = snr_val_librosa

    score_dict['lsd_pr'] = lsd_val
    score_dict['lsd_lbr'] = lsd_val_librosa

    x_pr = P.flatten()

    data = eval_single_MOS(x_pr, fs, fs)
    score_dict['P808_MOS pr'] = data['P808_MOS']
    score_dict['SIG_raw pr'] = data['SIG_raw']
    score_dict['BAK_raw pr'] = data['BAK_raw']
    score_dict['OVRL pr'] = data['OVRL']
    score_dict['SIG pr'] = data['SIG']
    score_dict['BAK pr'] = data['BAK']

    data = eval_single_MOS(Y.flatten(), fs, fs)
    score_dict['P808_MOS hr'] = data['P808_MOS']
    score_dict['SIG_raw hr'] = data['SIG_raw']
    score_dict['BAK_raw hr'] = data['BAK_raw']
    score_dict['OVRL hr'] = data['OVRL']
    score_dict['SIG hr'] = data['SIG']
    score_dict['BAK hr'] = data['BAK']

    data = eval_single_MOS(X.flatten(), fs, fs)
    score_dict['P808_MOS lbr'] = data['P808_MOS']
    score_dict['SIG_raw lbr'] = data['SIG_raw']
    score_dict['BAK_raw lbr'] = data['BAK_raw']
    score_dict['OVRL lbr'] = data['OVRL']
    score_dict['SIG lbr'] = data['SIG']
    score_dict['BAK lbr'] = data['BAK']

    return score_dict

# ----------------------------------------------------------------------------
def upsample(x_lr, r):

  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)

  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)

  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp


# ----------------------------------------------------------------------------
def get_spectrum(x, n_fft=2048):
  S = librosa.stft(x, n_fft = n_fft)
  p = np.angle(S)
  #S = np.log1p(np.abs(S))

  S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)

  return S_dB

def save_spectrum(S, sr, hop_length, outfile='spectrogram.png', type = "high resolution"):
    # Create a smaller figure with reduced size
    plt.figure(figsize=(6, 3))  # Adjust the figure size for smaller paper size

    # Plot the spectrogram with larger labels
    librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.yticks(ticks=np.arange(0, sr // 2 + 1, sr//8),
               labels=[f'{x / 1000:.1f}' for x in np.arange(0, sr // 2 + 1, sr//8)])

    # Add a color bar with larger font size
    cbar = plt.colorbar(format='%+2.0f dB')
    cbar.ax.tick_params(labelsize=15)  # Adjust the color bar tick labels size

    # Set title and axis labels with larger font size
    #plt.title('Spectrogram', fontsize=16)
    plt.title(type, fontsize=15)
    plt.xlabel('Time (s)', fontsize=15)
    plt.ylabel('Frequency (kHz)', fontsize=15)

    # Adjust tick parameters for both axes
    plt.tick_params(axis='both', which='major', labelsize=15)  # Major ticks
    plt.tick_params(axis='both', which='minor', labelsize=15)  # Minor ticks

    # Use tight layout for better spacing and save the figure
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')  # High DPI for better quality
    plt.close()

# Example usage
# save_spectrum(S, sr, hop_length, outfile='spectrogram.png')

def spline_up(x_lr, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)
    x_sp = interpolate.splev(i_hr, f)
    return x_sp

def eval_snr_lsd(generator, val_loader, model_path):

    if torch.cuda.is_available() :
        print("CUDA!")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #device = torch.device('cpu')
    state_dict = torch.load(model_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()

    generator.to(device)

    Y = []
    P = []
    X = []

    lsd_val_avg = []
    snr_val_avg = []

    lsd_val_avg_librosa = []
    snr_val_avg_librosa = []

    for i, (lr_sound, hr_sound) in enumerate(val_loader, 0):

        hr_sound, lr_sound = hr_sound.to(device).float(), \
                             lr_sound.to(device).float()

        fake_patches = generator(lr_sound).detach()

        P1 = fake_patches.cpu().numpy().flatten()
        Y1 = hr_sound.cpu().numpy().flatten()
        X1 = lr_sound.cpu().numpy().flatten()

        P.append(P1)
        Y.append(Y1)
        X.append(X1)

        #print(get_snr (P1, Y1), get_snr(X1, Y1))
        lsd = get_lsd(P1, Y1, n_fft = 2048)
        snr = get_snr (P1, Y1)

        print("lsd_snr", lsd, snr)

        lsd_val_avg.append( lsd)
        snr_val_avg.append( snr)

        lsd_val_avg_librosa.append(get_lsd(X1, Y1, n_fft=2048))
        snr_val_avg_librosa.append(get_snr(X1, Y1))

    Y = np.concatenate(Y)
    P = np.concatenate(P)
    X = np.concatenate(X)

    lsd_val = get_lsd(P, Y, n_fft=2048)
    snr_val = get_snr(P, Y)

    lsd_val_librosa = get_lsd(X, Y, n_fft=2048)
    snr_val_librosa = get_snr(X, Y)

    return lsd_val, snr_val, lsd_val_librosa, snr_val_librosa, np.mean(lsd_val_avg), np.mean(snr_val_avg)


