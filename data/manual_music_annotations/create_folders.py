from email.mime import audio
import os
import shutil

audios_list='yamnet_music_0.2.txt'
cwd=os.path.abspath(os.getcwd())

audio_path=os.path.join('/'.join(cwd.split('/')[:-1]),'OCEAN_new_structure/data/audio/')

out_audios_path=os.path.join(cwd,'audio')

if not os.path.exists(out_audios_path):
    os.mkdir(out_audios_path)

with open (audios_list) as file:
    files=file.read().splitlines()

for file in files:
    shutil.copy2(os.path.join(audio_path,file+'.wav'), out_audios_path)

