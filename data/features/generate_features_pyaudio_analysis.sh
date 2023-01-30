#!/bin/bash

NAME=pyaudio_features
mkdir $NAME
AUDIO_DIR='/home/gbarchi/Documentos/Trust/Personality/OCEAN_new_structure/data/audio'

python ../../src/extract_pyaudio_features.py $NAME/pyaudio_all_audio_duration.csv --files_path $AUDIO_DIR
