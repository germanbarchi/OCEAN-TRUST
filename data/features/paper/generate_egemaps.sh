
#!/bin/bash

NAME=data/features/paper/egemaps
mkdir $NAME
AUDIO_DIR='/home/gbarchi/Documentos/Trust/Personality/OCEAN_new_structure/data/audio'

echo 'Computing: egemaps_full_audio'
python src/extract_egemaps.py data/features/paper/egemaps/egemaps_full_audio.csv --files_path $AUDIO_DIR --normalize False --speech_ratio False --n_jobs -1

echo 'Computing: egemaps_speech'
python src/extract_egemaps.py data/features/paper/egemaps/egemaps_speech.csv --files_path silero_VAD/silero_speech --normalize False --speech_ratio False --n_jobs -1

echo 'Computing: egemaps_no_speech'
python src/extract_egemaps.py data/features/paper/egemaps/egemaps_no_speech.csv --files_path silero_VAD/silero_no_speech --normalize False --speech_ratio False --n_jobs -1

echo 'Computing: egemaps_normalized_full_audio'
python src/extract_egemaps.py data/features/paper/egemaps/egemaps_normalized_full_audio.csv --files_path $AUDIO_DIR --normalize True --norm_method min_max --speech_ratio False --n_jobs -1

echo 'Computing: egemaps_normalized_speech'
python src/extract_egemaps.py data/features/paper/egemaps/egemaps_normalized_speech.csv --files_path silero_VAD/silero_speech --normalize True --norm_method min_max --speech_ratio False --n_jobs -1
