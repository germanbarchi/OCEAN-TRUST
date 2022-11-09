import glob 
import numpy as np
from pathlib import Path
import sys 
import itertools

dir=Path(__file__)
parent=str(dir.parent.parent)
sys.path.append(parent)

from src import toolbox
from IPython import embed
import scipy.io as sio 
import soundfile 
import os 

def main(audio_paths,speech_dir,no_speech_dir):
    
    speech_classes_index=[i for i in range (46)]+[51,53,54]
    speech_threshold=0.01
    for i in audio_paths:
        th=0.5
        filename
        while max_prediction > speech_threshold or (th<=0.01):              

            silero_timestamp,speech,no_speech,fs=toolbox.silero_timestamps(input,th)
            temp_path='temp/no_speech.wav'
            soundfile.write(temp_path,no_speech,fs)    
            predictions, yamnet_classes=toolbox.yamnet_classifier(temp_path)
            
            speech_classes_index=[predictions[i] for i in itertools.chain(range (46),[51,53,54])]
            
            max_prediction=yamnet_classes[np.argmax(speech_classes_index)]
            
            th-=0.01
        
        save_audio(speech,no_speech,fs)
  

def save_audio(speech,no_speech,fs,speech_dir,non_speech_dir):
    if not os.path.exists(speech_dir):
        os.mkdir(speech_dir)
    if not os.path.exists(non_speech_dir):
        os.mkdir(non_speech_dir)
    
    soundfile.write(speech_dir,speech,fs)  
    soundfile.write(no_speech_dir,no_speech,fs)


if __name__=='__main__':

   #audio_path=
    audio_paths=glob.glob(audio_path+'/*/*')
    speech_dir=os.path.join(parent,data/silero+yamnet/)
    no_speech_dir=
    main(audio_paths,speech_dir,no_speech_dir)