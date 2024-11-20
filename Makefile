SINGLE_SPEAKER_DIR = /data/VCTK-Corpus/wav48/p225

# Patch generation parameters
SCA=4
TR_DIM=8192
TR_STR=4096
VA_DIM=8192
VA_STR=4096
SR=48000

# ----------------------------------------------------------------------------
all:
	make \
		/data/vctk/speaker1/vctk-speaker1-train.$(SCA).$(SR).$(TR_DIM).$(TR_STR).h5 \
		/data/vctk/speaker1/vctk-speaker1-val.$(SCA).$(SR).$(VA_DIM).$(VA_STR).h5

# ----------------------------------------------------------------------------
# Create dataset for one speaker

/data/vctk/speaker1/vctk-speaker1-train.%.$(SR).$(TR_DIM).$(TR_STR).h5: /data/vctk/speaker1/speaker1-train-files.txt
	python /src/prep_vctk_short.py \
		--file-list $< \
		--in-dir $(SINGLE_SPEAKER_DIR) \
		--out $@.tmp \
		--scale $* \
		--sr $(SR) \
		--dimension $(TR_DIM) \
		--stride $(TR_STR) \
		--low-pass
	mv $@.tmp $@

/data/vctk/speaker1/vctk-speaker1-val.%.$(SR).$(VA_DIM).$(VA_STR).h5: /data/vctk/speaker1/speaker1-val-files.txt
	python /src/prep_vctk_short.py \
		--file-list $< \
		--in-dir $(SINGLE_SPEAKER_DIR) \
		--out $@.tmp \
		--scale $* \
		--sr $(SR) \
		--dimension $(VA_DIM) \
		--stride $(VA_STR) \
		--low-pass
	mv $@.tmp $@


