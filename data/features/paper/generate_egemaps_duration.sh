#!/bin/bash

NAME=data/features/paper/egemaps_duration
mkdir $NAME
AUDIO_DIR='/home/gbarchi/Documentos/Trust/Personality/OCEAN_new_structure/data/audio'

for i in $(seq 1 2 15)
do 
    # generate egemaps with random samples

    python src/extract_egemaps.py $NAME/egemaps_full_audio_duration_${i}_random.csv --files_path $AUDIO_DIR --duration $i --random_sampling True --blacklist data/lists/blacklist.txt --speech_ratio False
done