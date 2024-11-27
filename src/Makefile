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

# -------------------
# prepare noisy test set
# -------------------
prepare_noisy_test:

	python3 noisy_test_audio_synthesizer.py  \
        --wav_file_list ../data/vctk/multispeaker/val-files.txt \
        --sr 16000 \
        --noisydir ../data/noise/DEMND-Corpus/ \
        --snr 10 \
        --target_level_lower -35 \
        --target_level_upper -15 \
        --output_dir_audio ../data/test \
        --in_dir ../data/vctk \
        --randomize_snr True

# -------------------------
run_evaluation_scores:

	python3 eval_scores.py eval \
	    --logname logs/multispeaker/sr16000/Alt5/multispeaker.r_4.gan.b128.sr_16000.generator_autoencoder_gan.epoch_210.pth\
        --wav_file_list ../data/vctk/multispeaker/val-files.txt \
        --model gan_multispeaker \
        --sr 16000 \
        --noisydir ../data/noise/DEMND-Corpus/ \
        --snr 10 \
        --target_level_lower -35 \
        --target_level_upper -15 \
        --output_dir_audio ../data/test \
        --in_dir ../data/vctk \
        --randomize_snr True \
        --csv_path ../results/MOS/scores_gan_Alt5.csv

# -------------------------
run_evaluation_samples:

	python3 test_other_audio.py \
		--model gan_multispeaker \
        --output ../data/examples/Denoised \
        --data ../data/examples \
        --sr 16000 \
        --logname logs/multispeaker/sr16000/gan_alt_5/multispeaker.r_4.gan.b128.sr_16000.generator_autoencoder_gan.epoch_210.pth

# -------------------------
run_plt_loss_gan:

	python3 plt_loss.py \
        --model gan_alt_5_multispeaker \
        --sr 16000

# -------------------------
run_hist_16_multispeaker_scores:

	python3 plt_hist_thesis_16.py \
	    --model gan \
	    --sr 16000 \
	    --speaker multi
