SINGLE_SPEAKER_DIR=../VCTK-Corpus/wav48/p225

# patch generation
TR_DIM=-1
TR_DIM=8192
TR_STR=4096
VA_DIM=-1
VA_DIM=8192
VA_STR=4096
SR=16000
SCA=4


# fraction of patches to keep
SAM=0.25


# -------------------
# train models
# -------------------
patches:
	make \
	run_training

run_training_gan_16_r_4_multispeaker:
	 python3 run_gan.py train \
		--train ../data/vctk/multispeaker/vctk-noise-train.4.16000.8192.8192.0.25.h5 \
		--val ../data/vctk/multispeaker/vctk-noise-val.4.16000.8192.8192.0.25.h5 \
		--e 400 \
		--batch_size 8 \
		--lr 0.0001\
		--logname multispeaker \
		--model gan \
		--r 4 \
		--layers 4 \
		--pool_size 2 \
		--strides 2 \
		--sr 16000 \
		--full false \
		--noise_dir ../data/noise/DEMND-Corpus/ \
		--snr 10

# -------------------
# evaluate models on test set
# -------------------

run_evaluation_gan_4_multispeaker:

	python3 run_gan.py eval \
	    --val ../data/vctk/multispeaker/vctk-noise-val.4.4000.8192.8192.0.25.h5 \
        --logname logs/sr4000/Alt5/singlespeaker.r_4.gan.b128.sr_4000.generator_autoencoder_gan.epoch_180.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model gan \
        --sr 4000 \
        --noise_dir ../data/noise/DEMND-Corpus/ \
        --speaker multi \
        --snr 10

run_evaluation_gan_16_multispeaker:

	python3 run_gan.py eval \
	    --val ../data/vctk/multispeaker/vctk-noise-val.4.16000.8192.8192.0.25.h5 \
        --logname logs/sr16000/Alt5/singlespeaker.r_4.gan.b128.sr_16000.generator_autoencoder_gan.epoch_180.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model gan \
        --sr 16000 \
        --noise_dir ../data/noise/DEMND-Corpus/ \
        --speaker multi \
        --snr 10


