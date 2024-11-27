import os
import argparse
from random import shuffle

import librosa
import soundfile as sf

import numpy as np
from audiolib import segmental_snr_mixer


# -----------------------------------------------------------------
def make_parser():

  parser = argparse.ArgumentParser()
  parser.add_argument('--func', default=synthesize_audio)
  parser.add_argument('--randomize_snr', default=True)
  parser.add_argument('--noisydir', type=str)
  parser.add_argument('--wav_file_list', type=str)
  parser.add_argument('--output_dir_audio', type=str)
  parser.add_argument('--sr', type=int, default=16000)
  parser.add_argument('--target_level_lower', type=int, default=-35)
  parser.add_argument('--target_level_upper', type=int, default=-15)
  parser.add_argument('--snr_lower', type=int, default=0)
  parser.add_argument('--snr_upper', type=int, default=40)
  parser.add_argument('--snr', type=int, default=10)
  parser.add_argument('--in_dir', type=str)

# - snr_upper: Upper bound for SNR required (default: 40 dB)
#  --wav_file_list.. / data / vctk / multispeaker / val - files.txt \
#  - -sr  16000 \
#  - -noise_dir.. / data / noise / DEMND - Corpus / \
#  --snr  10 \
#  --target_level_lower - 35 \
#  - -target_level_upper - 15 \
#  - -output_dir_audio.. / data / test /

  return parser

def synthesize_audio(args):

    file_extensions = [".wav", ".mp3"]
    with open(args.wav_file_list) as f:

        for line in f:

            filename = line.strip()
            ext = os.path.splitext(filename)[1]

            if ext in file_extensions:

                #print(filename.split("..")[-1])

                #file_path = os.path.join(args.in_dir, filename.split("..")[-1])
                file_path = args.in_dir + filename.split("..")[-1]

                #print(file_path)
                #input()

                clean_audio, fs = librosa.load(file_path, sr=args.sr)

                noisy_speech = build_noisy_audio(clean_audio, args)

                outname_audio = args.output_dir_audio + "/" + filename.split("..")[-1].split("/")[-1].split(".")[0]

                #print("outname audio", outname_audio)
                #input()

                sf.write(outname_audio + '.noise.wav', noisy_speech, fs)
                sf.write(outname_audio + '.clean.wav', clean_audio, fs)

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

def main():

  parser = make_parser()
  args = parser.parse_args()
  args.func(args)

if __name__ == '__main__':
  main()

