import torch
from random import shuffle

from calculate_snr_lsd import get_lsd, get_snr
from audiolib import segmental_snr_mixer



#from torchmetrics.audio import SignalImprovementMeanOpinionScore


import os

import librosa
import numpy as np
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

class ComputeScore:
    def __init__(self, primary_model_path, p808_model_path) -> None:
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)

    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length,
                                                  n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):

        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, aud, input_fs,  sampling_rate, is_personalized_MOS = False):

        fs = SAMPLING_RATE
        if input_fs != fs:
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs)
        else:
            audio = aud

        len_samples = int(INPUT_LENGTH * fs)

        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples): int((idx + INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis, :]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :,
                                  :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        clip_dict = dict()
        clip_dict['num_hops'] = num_hops
        clip_dict['OVRL_raw'] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict['SIG_raw'] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict['BAK_raw'] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict['OVRL'] = np.mean(predicted_mos_ovr_seg)
        clip_dict['SIG'] = np.mean(predicted_mos_sig_seg)
        clip_dict['BAK'] = np.mean(predicted_mos_bak_seg)
        clip_dict['P808_MOS'] = np.mean(predicted_p808_mos)
        return clip_dict

# ----------------------------------------------------------------------------
def eval_single_MOS(sample, sr, sr_target):
    """
    Function to compute DNSMOS for a single audio sample.

    Args:
    - sample_path (str): Path to the audio sample.
    - args: Arguments containing MOS settings like personalized_MOS and csv_path.

    Returns:
    - The MOS score for the sample.
    """

    # Set model paths
    p808_model_path = os.path.join('../DNSMOS', 'DNSMOS', 'model_v8.onnx')

    primary_model_path = os.path.join('../DNSMOS','DNSMOS', 'sig_bak_ovr.onnx')

    # Instantiate the ComputeScore class
    compute_score = ComputeScore(primary_model_path, p808_model_path)

    # Define necessary parameters for DNSMOS computation
    desired_fs = SAMPLING_RATE  # Assuming this is defined elsewhere

    # Compute MOS score for the single sample
    with ThreadPoolExecutor() as executor:

            future = executor.submit(compute_score, sample, sr, sr_target)

            data = future.result()
            #print(data['P808_MOS'])  # Display the result

    return data

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

    x_lr_upsampled = build_noisy_audio(x_hr, args)

    if fs == 16000:
        n_fft = 2048
    elif fs == 48000:
        n_fft = 3*2048
    else:
        n_fft = 2048

    pad_length = args.patch_size - (x_hr.shape[0] % args.patch_size)
    x_hr = np.pad(x_hr, (0, pad_length), 'constant', constant_values=(0, 0))

    # Downscale signal
    #x_lr = decimate(x_hr, args.r)
    #x_lr_upsampled = upsample(x_lr, args.r)

    n_patches = x_lr_upsampled.shape[0]//args.patch_size

    #x_lr_librosa = librosa.resample(x_lr, orig_sr=fs // args.r, target_sr=fs)

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

    #print("snr pr", snr_val)
    #print("snr noisy", snr_val_librosa)

    score_dict = dict()

    score_dict['filename'] = name
    score_dict['snr_pr'] = snr_val
    score_dict ['snr_noisy'] = snr_val_librosa

    score_dict['lsd_pr'] = lsd_val
    score_dict['lsd_noisy'] = lsd_val_librosa

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
    score_dict['P808_MOS noisy'] = data['P808_MOS']
    score_dict['SIG_raw noisy'] = data['SIG_raw']
    score_dict['BAK_raw noisy'] = data['BAK_raw']
    score_dict['OVRL noisy'] = data['OVRL']
    score_dict['SIG noisy'] = data['SIG']
    score_dict['BAK noisy'] = data['BAK']

    return score_dict


# ----------------------------------------------------------------------------
def get_spectrum(x, n_fft=2048):
  S = librosa.stft(x, n_fft = n_fft)
  p = np.angle(S)
  #S = np.log1p(np.abs(S))

  S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)

  return S_dB

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

def get_noisefilelist(args):

    noise_files = []

    for root, dirs, files in os.walk(args.noisydir):
        for file in files:
            if file.endswith("wav"):
#
                noise_files.append(os.path.join(root, file))

    return noise_files

def build_noisy_audio(clean_audio, args):
    '''Construct an audio signal from source files'''

    get_noisefilelist(args)

    source_files = get_noisefilelist(args)

    shuffle(source_files)
    idx = np.random.randint(0, np.size(source_files))
    noise, fs_noise = librosa.load(source_files[idx], sr=args.sr)

    if len(noise) < len(clean_audio):
        noise = np.tile(noise, int(np.ceil(len(clean_audio) / len(noise))))

    noise = noise[:len(clean_audio)]

    # mix clean speech and noise
    # if specified, use specified SNR value

    if not args.randomize_snr:   #params['randomize_snr']:
        snr = args.snr #params['snr']
    # use a randomly sampled SNR value between the specified bounds
    else:
        snr = np.random.randint(args.snr_lower, args.snr_upper)

    clean, noisenewlevel, noisyspeech, noisy_rms_level = \
        segmental_snr_mixer(args, clean=clean_audio, noise=noise, snr=snr)

    return noisyspeech