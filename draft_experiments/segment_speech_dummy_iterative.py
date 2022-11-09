
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
import matplotlib.pyplot as plt

def main(input,th):
    th_array=[]
    pred_array=[]
    speech_threshold=0.01
    max_prediction=1
    while max_prediction > speech_threshold or (th<=0.05):              

        silero_timestamp,speech,no_speech,fs=toolbox.silero_timestamps(input,th)
        temp_path='temp/no_speech_th_'+str(th)+'.wav'
        soundfile.write(temp_path,no_speech,fs)    
        predictions, yamnet_classes=toolbox.yamnet_classifier(temp_path)
        
        speech_predictions=[predictions[i] for i in itertools.chain(range (46),[51,53,54])]
        
        max_prediction=np.max(speech_predictions)
        max_class=yamnet_classes[np.argmax(speech_predictions)]
        
        th_array.append(th)
        pred_array.append(max_prediction)
        
        print ('silero_threshold: %2f\npred: %f, class: %s' % (th,max_prediction,max_class))
        th-=0.01
        
        savefig(th_array,pred_array)

    print ('final_pred: %f, final_class: %s' % (max_prediction,max_class))

def savefig(th,pred):
    
    plt.plot(th,pred)
    plt.ylabel('max speech classes prediction')
    plt.xlabel('Silero threshold')
    plt.title('non-speech segments max predictions of speech related content')
    plt.savefig('silero+yamnet_segmentation',dpi=300,transparent=False,facecolor='white')

if __name__=='__main__':

    audio_paths='/home/gbarchi/Documentos/Trust/OCEAN-TRUST/draft_experiments/-4J4xkfN5cI.002.wav'
    
    main(audio_paths,0.5)