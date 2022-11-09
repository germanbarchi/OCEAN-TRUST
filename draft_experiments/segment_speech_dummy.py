
import glob 
import numpy as np
from pathlib import Path
import sys 

dir=Path(__file__)
parent=str(dir.parent.parent)
sys.path.append(parent)

from src import toolbox
from IPython import embed
import scipy.io as sio 
import soundfile 
import itertools 

def main(input,th):

    silero_timestamp,speech,no_speech,fs=toolbox.silero_timestamps(input,th)
    temp_path='temp/no_speech.wav'
    soundfile.write(temp_path,no_speech,fs)    
    predictions, yamnet_classes=toolbox.yamnet_classifier(temp_path)
    
    speech_classes_index=[predictions[i] for i in itertools.chain(range (46),[51,53,54])]
    
    print(yamnet_classes[np.argmax(speech_classes_index)])

if __name__=='__main__':

    audio_paths='/home/gbarchi/Documentos/Trust/OCEAN-TRUST/experiments/-4J4xkfN5cI.002.wav'
    
    main(audio_paths,0.5)