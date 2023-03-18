
#!/bin/bash

NAME_LEC='data/features/Trust-V2_dataset/egemaps+sr_lecture'
NAME_FREE='data/features/Trust-V2_dataset/egemaps+sr_free'
AUDIO_DIR_FREE='data/features/Trust-V2_dataset/free_speech_paths.txt'
AUDIO_DIR_LEC='data/features/Trust-V2_dataset/lecture_speech_paths.txt'

echo 'Computing: egemaps_full_audio_lecture'
mkdir $NAME_LEC
python src/extract_egemaps.py ${NAME_LEC}/TrustV2_lecture-egemaps+sr-full_audio.csv --paths_list $AUDIO_DIR_LEC --normalize False --speech_ratio True --n_jobs 6

echo 'Computing: egemaps_full_audio_free'
mkdir $NAME_FREE
python src/extract_egemaps.py ${NAME_FREE}/TrustV2_free-egemaps+sr-full_audio.csv --paths_list $AUDIO_DIR_FREE --normalize False --speech_ratio True --n_jobs 6
