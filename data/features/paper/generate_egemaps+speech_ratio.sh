
#!/bin/bash

NAME=data/features/paper/egemaps+duration
mkdir $NAME
AUDIO_DIR='/home/gbarchi/Documentos/Trust/Personality/OCEAN_new_structure/data/audio'

echo 'Computing: egemaps_full_audio'
python src/extract_egemaps.py ${NAME}/egemaps_full_audio.csv --files_path $AUDIO_DIR --normalize False --speech_ratio True --n_jobs 6

echo 'Computing: egemaps_speech'
python src/extract_egemaps.py ${NAME}/egemaps_speech.csv --files_path silero_VAD/silero_speech --normalize False --speech_ratio True --n_jobs 6

echo 'Computing: egemaps_no_speech'
python src/extract_egemaps.py ${NAME}/egemaps_no_speech.csv --files_path silero_VAD/silero_no_speech --normalize False --speech_ratio True --n_jobs 6

echo 'Computing: egemaps_normalized_full_audio'
python src/extract_egemaps.py ${NAME}/egemaps_normalized_full_audio.csv --files_path $AUDIO_DIR --normalize True --norm_method min_max --speech_ratio True --n_jobs 6

echo 'Computing: egemaps_normalized_speech'
python src/extract_egemaps.py ${NAME}/egemaps_normalized_speech.csv --files_path silero_VAD/silero_speech --normalize True --norm_method min_max --speech_ratio True --n_jobs 6
