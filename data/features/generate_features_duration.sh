#!/bin/bash

NAME=features_duration_sr_fixed
mkdir $NAME
AUDIO_DIR='/home/gbarchi/Documentos/Trust/Personality/OCEAN_new_structure/data/audio'

for i in $(seq 13 2 15)
do 
    #python ../../src/extract_egemaps.py $NAME/egemaps_ns_duration_$i.csv --files_path ../../silero_VAD/silero_no_speech --duration $i
    #python ../../src/extract_egemaps.py $NAME/egemaps_s_duration_$i.csv --files_path ../../silero_VAD/silero_speech --duration $i
    #python ../../src/extract_egemaps.py $NAME/egemaps_all_audio_duration_$i.csv --files_path $AUDIO_DIR --duration $i

    # generate egemaps with random samples

    python ../../src/extract_egemaps.py $NAME/egemaps_all_audio_duration_${i}_random.csv --files_path $AUDIO_DIR --duration $i --random_sampling True --blacklist ../lists/blacklist.txt --speech_ratio True
done