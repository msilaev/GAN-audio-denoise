import os
import gc
from panns_inference import AudioTagging

import matplotlib.pyplot as plt
import soundfile as sf

import argparse

from models.audiounet import AudioUNet

from models.gan import Generator, Discriminator, \
    discriminator_loss, generator_loss, BCEWithSquareLoss

from calculate_snr_lsd import get_lsd, get_snr

#from models.gan_simple import SimpleDiscriminator, SimpleUNetGenerator, \
#    WaveGANDiscriminator, weights_init, WaveGANGenerator

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from dataset_batch_norm import BatchData
from models.metrics_calc import get_scores
import torch.optim as optim
#from torchinfo import summary
import numpy as np

import pandas as pd

# -----------------------------------------------------------------
def make_parser():

  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(title='Commands')

  eval_parser = subparsers.add_parser('eval')
  eval_parser.set_defaults(func=eval)
  eval_parser.add_argument('--logname', required=True)
  eval_parser.add_argument('--wav_file_list')
  eval_parser.add_argument('--sr',  type=int, default=48000)
  eval_parser.add_argument('--model',  choices=('gan_multispeaker', 'audiounet_multispeaker'))
  eval_parser.add_argument('--patch_size', type=int, default=8192)

  eval_parser.add_argument('--csv_path', type=str)

  eval_parser.add_argument('--randomize_snr', default=True)
  eval_parser.add_argument('--noisydir', type=str)

  eval_parser.add_argument('--output_dir_audio', type=str)

  eval_parser.add_argument('--target_level_lower', type=int, default=-35)
  eval_parser.add_argument('--target_level_upper', type=int, default=-15)
  eval_parser.add_argument('--snr_lower', type=int, default=0)
  eval_parser.add_argument('--snr_upper', type=int, default=40)
  eval_parser.add_argument('--snr', type=int, default=10)
  eval_parser.add_argument('--in_dir', type=str)

  return parser


def eval(args):

  if args.model in ["gan_multispeaker"]:

    model = Generator(layers=5)

  elif args.model in ["audiounet_multispeaker"]:

    model = AudioUNet(layers=4)

  model.eval()

  checkpoint_root = args.logname

  if args.wav_file_list:
    with open(args.wav_file_list) as f:

      rows = []

      file_exists = False

      for line in f:
        print(line)

        score_dict = \
              get_scores(model, '../data/vctk'+line.strip().split("..")[1],
                                                       line.strip().split("..")[1],
                         checkpoint_root, args)

        print(score_dict)

        rows.append(score_dict)
          #model, wav, name, model_path, args

        # Convert the current row(s) to a DataFrame
        df = pd.DataFrame([score_dict])

        # Append DataFrame to CSV after each step
        df.to_csv(args.csv_path, mode='a', header=not file_exists, index=False)

        # Set file_exists to True after first write
        file_exists = True

    #df = pd.DataFrame(rows)
    #if args.csv_path:
    #    csv_path = args.csv_path
    #    df.to_csv(csv_path)
    #else:
    #    print(df.describe())

def main():

  torch.cuda.empty_cache()
  gc.collect()

  parser = make_parser()
  args = parser.parse_args()
  args.func( args)

if __name__ == '__main__':
  main()