import glob
import numpy as np
from pathlib import Path
import sys,os
import tqdm 
dir=Path(__file__)
parent=str(dir.parent.parent)
sys.path.append(parent)

from src import toolbox
import soundfile

def get_no_speech(data_path,out_path,th):
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)   
    
    for path in tqdm.tqdm(data_path):
        filename=path.split('/')[-1]
        if (not filename in blacklist):
            silero_timestamp,speech,no_speech,fs=toolbox.silero_timestamps(path,th)
            soundfile.write(os.path.join(out_path,filename),no_speech,fs) 

if __name__=='__main__':

    audio_paths=glob.glob('/home/gbarchi/Documentos/Trust/Personality/OCEAN_new_structure/data/audio/*/*')   
    out='/home/gbarchi/Documentos/Trust/OCEAN-TRUST/data/silero'
    blacklist=['KRo-x2uoHUg.003.wav','KRo-x2uoHUg.002.wav']

    th_=[0.3,0.1]

    for th in tqdm.tqdm(th_):
        out_path=os.path.join(out,'no_speech_'+str(th))
        get_no_speech(audio_paths,out_path,th)

