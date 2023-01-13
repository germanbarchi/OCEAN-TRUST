#!/bin/bash

COMPLETAR DIRECTORIO DE AUDIOS

python extract_egemaps.py ../data/features/egemaps_all.csv --files_path ../data/>(directorio de audios) --normalize False 
python extract_egemaps.py ../data/features/egemaps_speech.csv --files_path ../data/silero/silero_speech --normalize False 
python extract_egemaps.py ../data/features/egemaps_no_speech.csv --files_path ../data/silero/silero_no_speech --normalize False 

python extract_egemaps.py ../data/features/egemaps_normalized_s.csv --files_path ../data/silero/silero_speech --normalize True --norm_method min_max
python extract_egemaps.py ../data/features/egemaps_normalized_ns.csv --files_path ../data/silero/silero_no_speech --normalize True --norm_method min_max

