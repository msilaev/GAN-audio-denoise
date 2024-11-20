#!/bin/sh

import os

import argparse
import numpy as np
import librosa

import matplotlib.pyplot as plt

# -------------------
# parser
# -------------------
def make_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--file_list',
        help='list of input wav files to process')
    parser.add_argument('--out_label', default='',
                        help='append label to output samples')
    parser.add_argument('--r', type=int, default=4, help='upscaling factor')
    parser.add_argument('--speaker', default='p225', help='help')

    parser.add_argument('--model')

    parser.add_argument('--sr')

    return parser

def get_snr(P, Y):
    # Compute L2 loss
    sqrt_l2_loss = (np.mean((P - Y) ** 2) )
    sqrn_l2_norm = (np.mean(Y ** 2))
    snr = 10 * np.log10(sqrn_l2_norm / (sqrt_l2_loss ))

    avg_snr = np.mean(snr)

    #plt.figure
    #plt.plot(P, marker='o',  linestyle='', label="HR")
    #plt.plot(Y, marker='o',  linestyle='', label="Pred")
    #plt.xlim([0, 100])
    #plt.ylim([-0.1, 0.1])
    #plt.grid(True)
    #plt.show()

    return avg_snr

def get_lsd(P, Y, n_fft):


    S_p = librosa.stft(P, n_fft=n_fft)
    S_y = librosa.stft(Y, n_fft=n_fft)

    ratio = ( ( np.abs(S_p) + 10**(-10) )/ ( np.abs(S_y) + 10**(-10) ))**2

    split_index = int(ratio.shape[0]*0.25)

    lsd = np.mean( np.sqrt( np.mean(  (np.log( ratio ))**2, axis = 0) ))

    lsd_low = np.mean( np.sqrt( np.mean(  (np.log( ratio [:split_index, :] ))**2, axis = 0) ))
    lsd_high = np.mean( np.sqrt( np.mean(  (np.log( ratio [split_index:, :] ))**2, axis = 0) ))


    return lsd #, lsd_low, lsd_high

########################################
def eval_snr_lsd(args):

  Y = []
  P = []
  X = []

  if args.file_list:

    with open(args.file_list) as f:
      for line in f:
        try:
          x_hr, x_pr, x_lr = load_wav(line.strip(), args)
          P.append(x_pr)
          Y.append(x_hr)
          X.append(x_lr)

        except EOFError:
          print('WARNING: Error reading file:', line.strip())

  Y = np.concatenate(Y)
  P = np.concatenate(P)
  X = np.concatenate(X)

  #return get_snr(P, Y)
  if args.sr == "48000":
      n_fft = 1 * 3 * 2048
  elif args.sr == "16000":
      n_fft = 1 * 1 * 2048

  return get_snr(P, Y), get_snr(X, Y), get_lsd(P, Y, n_fft), get_lsd(X, Y, n_fft)

########################################
def load_wav(wav, args):

    if args.sr == '16000':
        output_dir_audio = '../results/audio16/'
    elif args.sr == '48000':
        output_dir_audio = '../results/audio48/'

    outname_audio = output_dir_audio +  wav.split("/")[-1] + '.' + args.out_label + '.torch'

    x_lr, _ = librosa.load(outname_audio + '.r' + str(args.r) + '.lr.wav', sr= None)
    x_hr, _ = librosa.load(outname_audio +  '.hr.wav' , sr= None)
    x_pr, _ = librosa.load(outname_audio + '.r' + str(args.r)  + '.' + str(args.model)  +
                           '.pr.wav',  sr= None)
    return x_hr, x_pr, x_lr


def main():
    parser = make_parser()
    args = parser.parse_args()

    metric = eval_snr_lsd(args)
    #get_snr(P, Y), get_snr(X, Y), get_lsd(P, Y, n_fft), get_lsd(X, Y, n_fft)

    print(f"Upsampled to {args.sr} Hz with factor {args.r}, model simple, "
          f"SNR = {metric[1]}, LSD = {metric[3]} ")

    print(f"Upsampling to {args.sr} Hz with factor {args.r}, model {args.model}, "
          f"SNR = {metric[0]}, LSD = {metric[2]} ")

if __name__ == "__main__":
    main()
