import glob
import os 
import subprocess
import pathlib 

base_path=pathlib.Path(os.getcwd())

audio_path=str(base_path.parents[1].joinpath('Personality/OCEAN_new_structure/data/audio/*/*.wav'))
music_list=str(base_path.parents[0].joinpath('data/lists/yamnet_music_0.2.txt'))
audio_output=str(base_path.parents[0].joinpath('data/source_separation/separated_audios'))

audio_paths=glob.glob(audio_path)

with open (music_list,'r') as file:
    paths=file.read().splitlines()

audio_inputs=[abs_path for path in paths for abs_path in audio_paths if path in abs_path]

if not os.path.exists(audio_output):
    os.makedirs(audio_output)

for input in audio_inputs:
    print (input)
    subprocess.run(['demucs', '--two-stems=vocals','-o='+audio_output, input])