import os
import numpy as np
import h5py
import torch
import librosa
import soundfile as sf
from scipy import interpolate

from scipy.signal import decimate
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt
import random

# ----------------------------------------------------------------------------
def load_h5(h5_path):
  # load training data
  #print(h5_path)
  with h5py.File(h5_path, 'r') as hf:
    X = np.array(hf.get('data'))
    Y = np.array(hf.get('label'))

  return X, Y

def upsample_wav(model, wav, args, noise_files, epoch = None, model_path = None):

    if (model_path is not None):
        # Load the model
        device = torch.device('cpu')
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

    else:
        if torch.cuda.is_available():
            print("CUDA!")
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    # Load signal
    x_hr, fs = librosa.load(wav, sr=args.sr)
    #x_lr_t = spline_up(decimate(x_hr, args.r), args.r)
    # Pad to multiple of patch size to ensure model runs over entire sample

    pad_length = args.patch_size - (x_hr.shape[0] % args.patch_size)
    x_hr = np.pad(x_hr, (0, pad_length), 'constant', constant_values=(0, 0))

    # Downscale signal
    #x_lr = decimate(x_hr, args.r)
    #x_lr_1 = upsample(x_lr, args.r)

    x_lr_1 = add_noise(x_hr, noise_files, args)

    #x_lr = librosa.resample(x_lr, orig_sr=fs // args.r, target_sr=fs)
    #x_lr = x_lr[:len(x_lr) - (len(x_lr) % (2 ** (args.layers + 1)))]

    x_lr_1 = x_lr_1[:len(x_lr_1) - (len(x_lr_1) % (2 ** (args.layers + 1)))]

    x_lr_tensor_part = torch.tensor(x_lr_1.flatten(),
                                    dtype=torch.float32).unsqueeze(0).unsqueeze(2)

    # Upscale the low-res version using the model
    with torch.no_grad():

        #P = model(x_lr_tensor_part.to(device)).squeeze().numpy()
        P = model(x_lr_tensor_part.to(device)).squeeze().cpu().numpy()

    x_pr = P.flatten()

    # Save the audio and spectrum
    if fs == 4000:

        output_dir_spectrograms = 'logs/results/spectrograms4/'
        output_dir_audio = 'logs/results/audio4/'

    elif fs == 16000:

        output_dir_spectrograms = 'logs/results/spectrograms16/'
        output_dir_audio = 'logs/results/audio16/'


    if epoch is not None:

            outname_spectrograms = output_dir_spectrograms + wav.split('/')[-1] + \
                                   '.' + "epoch_" + str(epoch) + '.torch'

            outname_audio = output_dir_audio + wav.split('/')[-1] + \
                            '.' + "epoch_" + str(epoch) + '.torch'
    else:

            outname_spectrograms = output_dir_spectrograms + wav.split('/')[-1] + \
                                   '.' + args.out_label + '.torch'

            outname_audio = output_dir_audio + wav.split('/')[-1] + \
                            '.' + args.out_label + '.torch'

            sf.write(outname_audio + '.r' + str(args.r) + '.lr.wav', x_lr_1, fs)
            sf.write(outname_audio + '.hr.wav', x_hr, fs)

            save_spectrum(get_spectrum(x_hr, n_fft=1 * 2048), sr=fs, hop_length=2048 // 4,
                          outfile=outname_spectrograms + '.hr.png', type='high resolution')

            save_spectrum(get_spectrum(x_lr_1, n_fft=1 * 2048), sr=fs, hop_length=1 * 2048 // 4,
                          outfile=outname_spectrograms + '.r' + str(args.r) +
                                  '.sr' + str(args.sr) + '.spline.png', type='noisy')

    sf.write(outname_audio + '.r' + str(args.r) +
                 '.' + str(args.model) + '.pr.wav', x_pr, fs)

    save_spectrum(get_spectrum(x_pr, n_fft=1*2048), sr= fs, hop_length=2048//4,
                  outfile=outname_spectrograms + '.r' + str(args.r) +
                          '.' + str(args.model) +
                          '.sr' + str(args.sr) + '.pr.png', type = 'upsampled ' + str(args.model)  )

def add_noise(x, noise_files, args):

    noise_file = random.choice(noise_files)
    noise, noise_sr = librosa.load(noise_file, sr=args.sr)

    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    noise = noise[:len(x)]

    # Scale noise to achieve desired SNR
    snr = args.snr  # Desired Signal-to-Noise Ratio in dB
    signal_power = np.mean(x ** 2)
    noise_power = np.mean(noise ** 2)
    scaling_factor = np.sqrt(signal_power / (10 ** (float(args.snr) / 10) * noise_power))
    noise = noise * scaling_factor

    x_noisy = x + noise  # Add noise to the original signal

    return x_noisy

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
