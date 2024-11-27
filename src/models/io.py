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
# ---------------------------------------------------------------

def load_h5(h5_path):
  # load training data
  #print(h5_path)
  with h5py.File(h5_path, 'r') as hf:
    X = np.array(hf.get('data'))
    Y = np.array(hf.get('label'))

  return X, Y


def inference_wav_other_audio(model, wav, args, epoch = None, model_path = None):

    if torch.cuda.is_available():
            print("CUDA!")
            device = torch.device('cuda')
    else:
            device = torch.device('cpu')

    if (model_path is not None):
    #    # Load the model
    #    device = torch.device('cpu')
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

    model.to(device)

    # Load signal
    x_noisy, fs = librosa.load(wav, sr = args.sr)
    #x_lr_1 = x_noisy

    n_patches = x_noisy.shape[0]//args.patch_size

    #x_lr = librosa.resample(x_lr, orig_sr=fs // args.r, target_sr=fs)
    #x_lr = x_lr[:len(x_lr) - (len(x_lr) % (2 ** (args.layers + 1)))]

    P = []
    Y = []
    X = []

    with torch.no_grad():
        for i in range(0, n_patches, 1):

        #P = model(x_lr_tensor_part.to(device)).squeeze().numpy()
            lr_patch = np.array(x_noisy[i * args.patch_size : (i+1)* args.patch_size ])

            lr_patch_1 = np.array(x_noisy[i * args.patch_size: (i + 1) * args.patch_size])


            x_lr_tensor_part = torch.tensor(lr_patch.flatten(),
                                        dtype=torch.float32).unsqueeze(0).unsqueeze(2)

            x_pr_part = model(x_lr_tensor_part.to(device)).squeeze().cpu().numpy()

            P.append( x_pr_part)
            #Y.append(model(x_hr_tensor_part.to(device)).squeeze().cpu().numpy())

            X.append(lr_patch_1)

            #display_spectrum(get_mel_spectrum(x_pr_part, n_fft=1 * 2048), sr=args.sr, hop_length = 512,  type=wav)

    P = np.concatenate(P)
    X = np.concatenate(X)

    return P, X


def inference_wav(model, wav, args, epoch = None, model_path = None):

    if torch.cuda.is_available():
            print("CUDA!")
            device = torch.device('cuda')
    else:
            device = torch.device('cpu')

    if (model_path is not None):
    #    # Load the model
    #    device = torch.device('cpu')
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

    model.to(device)

    # Load signal
    x_hr, fs = librosa.load(wav, sr=int(args.sr))

    # Pad to multiple of patch size to ensure model runs over entire sample
    pad_length = args.patch_size - (x_hr.shape[0] % args.patch_size)
    x_hr = np.pad(x_hr, (0, pad_length), 'constant', constant_values=(0, 0))

    # Downscale signal
    x_lr = decimate(x_hr, args.r)
    x_lr_1 = upsample(x_lr, args.r)

    n_patches = x_lr_1.shape[0]//args.patch_size

    x_lr = librosa.resample(x_lr, orig_sr=fs // args.r, target_sr=fs)
    x_lr = x_lr[:len(x_lr) - (len(x_lr) % (2 ** (args.layers + 1)))]

    P = []
    Y = []
    X = []

    with torch.no_grad():
        for i in range(0, n_patches, 1):

        #P = model(x_lr_tensor_part.to(device)).squeeze().numpy()
            lr_patch = np.array(x_lr_1[i * args.patch_size : (i+1)* args.patch_size ])

            lr_patch_1 = np.array(x_lr[i * args.patch_size: (i + 1) * args.patch_size])

            hr_patch = np.array(x_hr[i * args.patch_size: (i+1) * args.patch_size ])

            x_lr_tensor_part = torch.tensor(lr_patch.flatten(),
                                        dtype=torch.float32).unsqueeze(0).unsqueeze(2)

            x_hr_tensor_part = torch.tensor(hr_patch.flatten(),
                                        dtype=torch.float32).unsqueeze(0).unsqueeze(2)

            x_pr_part = model(x_lr_tensor_part.to(device)).squeeze().cpu().numpy()

            P.append( x_pr_part)
            #Y.append(model(x_hr_tensor_part.to(device)).squeeze().cpu().numpy())
            Y.append(hr_patch)

            X.append(lr_patch_1)

            #display_spectrum(get_mel_spectrum(x_pr_part, n_fft=1 * 2048), sr=args.sr, hop_length = 512,  type=wav)

    P = np.concatenate(P)
    Y = np.concatenate(Y)
    X = np.concatenate(X)

    return P, Y, X


def upsample_wav(model, wav, args, epoch = None, model_path = None):

    P, Y, X = inference_wav(model, wav, args, epoch=epoch, model_path=model_path)

    snr_val = get_snr(P, Y)
    snr_val_x = get_snr(X, Y)

    print("snr pr", snr_val)
    print("snr librosa", snr_val_x)

    #print("r", args.r)
    #print("sr", args.sr)
    #print("patch size", args.patch_size)
    #input()

    x_pr = P.flatten()
    x_hr = Y.flatten()
    x_init = X.flatten()

    #x_pr = filter_artifacts(x_pr, args.sr)
    #plt.plot(x_pr, color = "blue")
    #plt.plot(x_hr, color = "green")
    #plt.plot(x_init, color = "red")
    #plt.show()
    #input()

    # Save the audio and spectrum
    if args.sr == 16000:

        output_dir_spectrograms = '../results/spectrograms16/samples/' + args.model+ '.sr_16.r_' + str(args.r) + "/"

        output_dir_audio = '../results/audio16/samples/' + args.model+ '.sr_16.r_' + str(args.r) + "/"

        output_dir_mel = '../results/mel16/samples/' + args.model+ '.sr_16.r_' + str(args.r) + "/"

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

            outname_mel = output_dir_mel + wav.split('/')[-1] + \
                            '.' + args.out_label + '.torch'

            outname_spectrograms = output_dir_spectrograms + wav.split('/')[-1]

            outname_audio = output_dir_audio + wav.split('/')[-1]

            outname_mel = output_dir_mel + wav.split('/')[-1]

            sf.write(outname_audio + '.r' + str(args.r) + '.lr.wav', x_init, args.sr)

            sf.write(outname_audio + '.hr.wav', x_hr, args.sr)

            save_spectrum(get_spectrum(x_hr, n_fft=1 * 2048), sr = args.sr, hop_length=2048 // 4,
                          outfile=outname_spectrograms + '.hr.png', type='high resolution')

            save_spectrum(get_spectrum(x_init, n_fft=1 * 2048), sr = args.sr, hop_length=1 * 2048 // 4,
                          outfile=outname_spectrograms + '.r' + str(args.r) +
                                  '.sr' + str(args.sr) + '.lr.png', type='low resolution')

            #save_spectrum(get_spectrum(x_lr_1, n_fft=1 * 2048), sr = args.sr, hop_length=1 * 2048 // 4,
            #              outfile=outname_spectrograms + '.r' + str(args.r) +
            #                      '.sr' + str(args.sr) + '.spline.png', type='upsampled, spline')

            #save_mel( get_spectrum(x_lr_1, n_fft = 1 * 2048), outfile = outname_mel + '.r' + str(args.r) +
            #                      '.sr' + str(args.sr) + '.pr.npy')

        sf.write(outname_audio + '.r' + str(args.r) +
                 '.' + str(args.model) + '.pr.wav', x_pr, args.sr)

        save_spectrum(get_spectrum(x_pr, n_fft=1*2048), sr = args.sr, hop_length=2048//4,
                  outfile=outname_spectrograms + '.r' + str(args.r) +
                          '.' + str(args.model) +
                          '.sr' + str(args.sr) + '.pr.png',
                      type = 'upsampled ' + str(args.model)  )

    if args.sr == 48000:

        #output_dir_spectrograms = 'logs/results/spectrograms48/'
        #output_dir_audio = 'logs/results/audio48/'

        output_dir_spectrograms = '../results/spectrograms48/samples/' + args.model+ '.sr_48.r_' + str(args.r) + "/"
        output_dir_audio = '../results/audio48/samples/' + args.model+ '.sr_48.r_' + str(args.r) + "/"

        # ' + args.model+ 'sr_48.r_' + str(args.r)

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

            outname_spectrograms = output_dir_spectrograms + wav.split('/')[-1]
            outname_audio = output_dir_audio + wav.split('/')[-1]

            sf.write(outname_audio + '.r' + str(args.r) + '.lr.wav', x_init, args.sr)
            sf.write(outname_audio + '.hr.wav', x_hr, args.sr)

            save_spectrum(get_spectrum(x_hr, n_fft=3 * 2048), sr = args.sr, hop_length=3*2048 // 4,
                          outfile=outname_spectrograms + '.hr.png', type='high resolution')

            save_spectrum(get_spectrum(x_init, n_fft=3 * 2048), sr = args.sr, hop_length=3 * 2048 // 4,
                          outfile=outname_spectrograms + '.r' + str(args.r) +
                                  '.sr' + str(args.sr) + '.lr.png', type='low resolution')

            #save_spectrum(get_spectrum(x_lr_1, n_fft=3 * 2048), sr = args.sr, hop_length=3 * 2048 // 4,
            #              outfile=outname_spectrograms + '.r' + str(args.r) +
            #                      '.sr' + str(args.sr) + '.spline.png', type='upsampled, spline')

        sf.write(outname_audio + '.r' + str(args.r) +
                 '.' + str(args.model) + '.pr.wav', x_pr, args.sr)

        save_spectrum(get_spectrum(x_pr, n_fft=3*2048), sr = args.sr, hop_length=3*2048//4,
                  outfile=outname_spectrograms + '.r' + str(args.r) +
                          '.' + str(args.model) +
                          '.sr' + str(args.sr) + '.pr.png',
                      type = 'upsampled ' + str(args.model)  )

def filter_artifacts(x_pr, sr):

    # Design a notch filter at 4 kHz
    Q = 30.0  # Quality factor (controls the bandwidth of the notch)

    freq = 2500  # Frequency to remove (4 kHz)
    b, a = signal.iirnotch(freq, Q, sr)
    x_pr = signal.filtfilt(b, a, x_pr)

    freq = 2000  # Frequency to remove (4 kHz)
    b, a = signal.iirnotch(freq, Q, sr)
    x_pr = signal.filtfilt(b, a, x_pr)

    freq = 3500  # Frequency to remove (4 kHz)
    b, a = signal.iirnotch(freq, Q, sr)
    x_pr = signal.filtfilt(b, a, x_pr)

    freq = 3000  # Frequency to remove (4 kHz)
    b, a = signal.iirnotch(freq, Q, sr)
    x_pr = signal.filtfilt(b, a, x_pr)

    freq = 4000  # Frequency to remove (4 kHz)
    b, a = signal.iirnotch(freq, Q, sr)
    x_pr = signal.filtfilt(b, a, x_pr)

    freq = 5000  # Frequency to remove (4 kHz)
    b, a = signal.iirnotch(freq, Q, sr)
    x_pr = signal.filtfilt(b, a, x_pr)

    freq = 6000  # Frequency to remove (4 kHz)
    b, a = signal.iirnotch(freq, Q, sr)
    x_pr = signal.filtfilt(b, a, x_pr)

    freq = 4500  # Frequency to remove (4 kHz)
    b, a = signal.iirnotch(freq, Q, sr)
    x_pr = signal.filtfilt(b, a, x_pr)

    freq = 5500  # Frequency to remove (4 kHz)
    b, a = signal.iirnotch(freq, Q, sr)
    x_pr = signal.filtfilt(b, a, x_pr)

    freq = 6500  # Frequency to remove (4 kHz)
    b, a = signal.iirnotch(freq, Q, sr)
    x_pr = signal.filtfilt(b, a, x_pr)

    freq = 7000  # Frequency to remove (4 kHz)
    b, a = signal.iirnotch(freq, Q, sr)
    x_pr = signal.filtfilt(b, a, x_pr)

    return x_pr

def upsample(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)

  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)

  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp

# -----------------------------------------------------------
def get_mel_spectrum(x, sr=16000, n_fft=2048, n_mels = 80):
    # Compute the Mel spectrogram
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft, n_mels=n_mels)

    # Convert to decibel scale (log scale) for better numerical stability and perceptual relevance
    S_dB = librosa.power_to_db(S, ref=np.max)

    return S

def get_spectrum(x, n_fft=2048):
  S = librosa.stft(x, n_fft = n_fft)
  p = np.angle(S)
  #S = np.log1p(np.abs(S))

  S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)

  return S_dB

def display_spectrum(S, sr, hop_length, type = "high resolution"):
    # Create a smaller figure with reduced size
    plt.figure(figsize=(5, 5))  # Adjust the figure size for smaller paper size

    # Set font sizes globally
    plt.rcParams.update({'font.size': 20})  # General font size
    plt.rcParams.update({'axes.titlesize': 20})  # Title font size
    plt.rcParams.update({'axes.labelsize': 20})  # X and Y label font size
    plt.rcParams.update({'legend.fontsize': 20})  # Legend font size
    plt.rcParams.update({'xtick.labelsize': 20})  # X tick label font size
    plt.rcParams.update({'ytick.labelsize': 20})  # Y tick label font size

    # Plot the spectrogram with larger labels
    librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.yticks(ticks=np.arange(0, sr // 2 + 1, sr // 8),
               labels=[f'{x / 1000:.1f}' for x in np.arange(0, sr // 2 + 1, sr // 8)])

    # Add a color bar with larger font size
    cbar = plt.colorbar(format='%+2.0f dB')
    #cbar.ax.tick_params(labelsize=15)  # Adjust the color bar tick labels size

    # Set title and axis labels with larger font size
    # plt.title('Spectrogram', fontsize=16)
    plt.title(type)
    plt.xlabel('Time (s)',)
    plt.ylabel('Frequency (kHz)')

    max_time = S.shape[1] * hop_length / sr  # Convert frames to seconds
    plt.xlim(0, max_time)  # Limit x-axis to the range of the data
    plt.xticks(ticks=[i for i in range(int(max_time) + 1) if i <= max_time])

    # Adjust tick parameters for both axes
    plt.tick_params(axis='both', which='major')  # Major ticks
    plt.tick_params(axis='both', which='minor')  # Minor ticks

    # Use tight layout for better spacing and save the figure
    plt.tight_layout()
    plt.show()

def save_mel(S, outfile='mel.npy'):

    np.save(outfile, S)

def save_spectrum(S, sr, hop_length, outfile='spectrogram.png', type = "high resolution"):
    # Create a smaller figure with reduced size
    plt.figure(figsize=(5, 5))  # Adjust the figure size for smaller paper size

    # Set font sizes globally
    plt.rcParams.update({'font.size': 20})  # General font size
    plt.rcParams.update({'axes.titlesize': 20})  # Title font size
    plt.rcParams.update({'axes.labelsize': 20})  # X and Y label font size
    plt.rcParams.update({'legend.fontsize': 20})  # Legend font size
    plt.rcParams.update({'xtick.labelsize': 20})  # X tick label font size
    plt.rcParams.update({'ytick.labelsize': 20})  # Y tick label font size

    # Plot the spectrogram with larger labels
    librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.yticks(ticks=np.arange(0, sr // 2 + 1, sr//8),
               labels=[f'{x / 1000:.1f}' for x in np.arange(0, sr // 2 + 1, sr//8)])

    # Add a color bar with larger font size
    cbar = plt.colorbar(format='%+2.0f dB')
    #cbar.ax.tick_params(labelsize=15)  # Adjust the color bar tick labels size

    # Set title and axis labels with larger font size
    #plt.title('Spectrogram', fontsize=16)
    #plt.title(type, fontsize=15)

    # Calculate the maximum time based on the number of frames in S
    max_time = S.shape[1] * hop_length / sr  # Convert frames to seconds
    plt.xlim(0, max_time)  # Limit x-axis to the range of the data
    plt.xticks(ticks=[i for i in range(int(max_time) + 1) if i <= max_time])

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (kHz)')

    # Adjust tick parameters for both axes
    plt.tick_params(axis='both', which='major')  # Major ticks
    plt.tick_params(axis='both', which='minor')  # Minor ticks

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

        #print("lsd_snr", lsd, snr)

        lsd_val_avg.append( lsd)
        snr_val_avg.append( snr)

        lsd_val_avg_librosa.append(get_lsd(X1, Y1, n_fft=2048))
        snr_val_avg_librosa.append(get_snr(X1, Y1))

#########################################
        x_pr = P1.flatten()
        x_hr = Y1.flatten()
        x_init = X1.flatten()

        #plt.plot(x_hr, color="blue")
       # plt.plot(x_hr, color="green")
        #plt.plot(x_pr, color="red")
        #plt.show()

############################################

    Y = np.concatenate(Y)
    P = np.concatenate(P)
    X = np.concatenate(X)

    lsd_val = get_lsd(P, Y, n_fft=2048)
    lsd_val_kuleshov = get_lsd_kuleshov(P, Y)

    snr_val = get_snr(P, Y)

    lsd_val_spline = get_lsd(X, Y, n_fft=2048)
    snr_val_spline = get_snr(X, Y)

    return lsd_val_kuleshov, lsd_val, snr_val, lsd_val_spline, snr_val_spline, np.mean(lsd_val_avg), np.mean(snr_val_avg)


