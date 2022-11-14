from email.mime import audio
import os
import shutil
import glob 
from IPython import embed 
from pathlib import Path 

audios_list='../lists/silero_valid_split_no_speech_greater_2sec.txt'
cwd=Path(os.getcwd())

audio_path=os.path.join(cwd.parents[2],'OCEAN_new_structure/data/silero/silero_no_speech/')

if not os.path.exists('audio'):
    os.mkdir('audio')

with open (audios_list) as file:
    files=file.read().splitlines()

for file in files:
    shutil.copy2(glob.glob(audio_path+'*/'+file)[0], 'audio')

