Audio Denoising using Generative Adversarial Networks
=====================================================

This repository contains PyTorch implementation of GAN approach to denosing.


## Installation

### Requirements

The model is implemented in Python 3.11.0 and uses several additional libraries which can be found in
environments.yaml


### Setup

To install this package, simply clone the git repo:

```
git clone ...
cd adversarial-denoising
conda env create -f environment.yaml
conda activate audio-enh-supervise
```

## Running the model

### Contents

The repository is structured as follows.

* `./src`: model source code
* `./data`: code to download the model data

### Retrieving data

The `./data` subfolder contains code for preparing the VCTK speech dataset with 
clean and noisy samples. VCTK dataset can be downloaded and unpacked by running
```
cd ./data/vctk
make
```
DEMAND noise dataset can be dowloaded by running 
```
cd ./data/noise
python load_dataset.py
```

Next, you must prepare the dataset for training:
you will need to create pairs of high and low resolution sound patches (typically, about 0.5s in length).
I have included a script called `prep_vctk_short.py` that does that. 

The output of the data preparation step are two `.h5` archives containing, respectively, the training and validation pairs of clean/noisy sound patches.
You can also generate these by running `make` in the corresponding directory, e.g.
```
cd ./data/vctk/multispeaker
make
```

The key parameters of datasets are the sampling rate SR (e.g. to 16000 or 48000) in the `/multispeaker/Makefile` file.


### Audio denoising tasks


### Training the model

Running supervised model is handled by the `src/run_supervised.py` script.
This script is launched as 

```
cd ./src:
make run_training;
```

Running GAN models is handled by the `src/run_gan.py` script.
This script is launched as 

```
cd ./src
make run_training_gan_16_r_4_multispeaker
```


On remote HPC machine using SLURM system the virtual environment is created and training is launched by the command 

```
sbatch run_train.sh
```

### Testing models

Trained models files should be put in folder `src/logs`. 

