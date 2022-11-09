
import glob
from os import mkdir 
import numpy as np
from pathlib import Path
import sys,os

dir=Path(__file__)
parent=str(dir.parent.parent)
sys.path.append(parent)

from src import toolbox
from IPython import embed
import scipy.io as sio 
import soundfile 
import itertools 
import matplotlib.pyplot as plt
import tqdm

def main(input,th,filename,no_speech_dir):
    th_array=[]
    pred_array=[]
    speech_threshold=0.01
    max_prediction=1
    
    while max_prediction > speech_threshold and (th>=0.05):              

        silero_timestamp,speech,no_speech,fs=toolbox.silero_timestamps(input,th)
        temp_path='temp/no_speech_temp.wav'
        soundfile.write(temp_path,no_speech,fs)    
        predictions, yamnet_classes=toolbox.yamnet_classifier(temp_path)
        
        speech_predictions=[predictions[i] for i in itertools.chain(range (46),[51,53,54])]
        
        max_prediction=np.max(speech_predictions)
        max_class=yamnet_classes[np.argmax(speech_predictions)]
        
        th-=0.01
        print('filename: %s' % filename)
        print('threshold: %2f\nmax_speech_pred: %2f' % (th,max_prediction))    
    
    savewavs(no_speech,fs,filename,no_speech_dir)

    print ('final_pred: %f, final_class: %s' % (max_prediction,max_class))

def savewavs(no_speech,fs,filename, no_speech_dir):   

    if not os.path.exists(no_speech_dir):
        os.mkdir(no_speech_dir)
    
    soundfile.write(os.path.join(no_speech_dir,filename),no_speech,fs)    

if __name__=='__main__':
    
    audio_paths=glob.glob('/home/gbarchi/Documentos/Trust/Personality/OCEAN_new_structure/data/audio/*/*')   
    no_speech_dir=os.path.join(parent,'data/no_speech_audios')
    
    for path in tqdm.tqdm(audio_paths):
        
        filename=path.split('/')[-1]
        
        if not os.path.exists(os.path.join(no_speech_dir,filename)):
        
            main(path,0.5,filename,no_speech_dir)