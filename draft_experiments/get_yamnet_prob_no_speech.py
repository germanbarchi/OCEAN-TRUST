
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
import pandas as pd 

def get_probs(input):
    
    df=pd.DataFrame()
    filename=path.split('/')[-1]
    silero_th=[0.5,0.3,0.1]
    th_tags=['pred_th_0.5','pred_th_0.3','pred_th_0.1']
    print(filename)
    for th in silero_th:

        silero_timestamp,speech,no_speech,fs=toolbox.silero_timestamps(input,th)

        temp_path='temp/no_speech_temp_th_'+str(th)+'.wav'
        soundfile.write(temp_path,no_speech,fs)  
        predictions, yamnet_classes=toolbox.yamnet_classifier(temp_path)
        
        speech_predictions=[predictions[i] for i in itertools.chain(range (46),[51,53,54])]
        
        max_prediction=np.max(speech_predictions)
        max_class=yamnet_classes[np.argmax(speech_predictions)]
        
        print('filename: %s' % filename)
        print('threshold: %2f\nmax_speech_pred: %2f' % (th,max_prediction))    
        tag=th_tags[silero_th.index(th)]
        df['filename']=filename,
        df[tag]=max_prediction

    print ('final_pred: %f, final_class: %s' % (max_prediction,max_class))

    return df

if __name__=='__main__':
    
    audio_paths=glob.glob('/home/gbarchi/Documentos/Trust/Personality/OCEAN_new_structure/data/audio/*/*')   
    out_df_path=os.path.join(parent,'data/metadata/no_speech_df_3.csv')
    results_df=pd.read_csv('/home/gbarchi/Documentos/Trust/OCEAN-TRUST/data/metadata/no_speech_df_2.csv')
    #results_df=pd.DataFrame()
    blacklist=['KRo-x2uoHUg.003.wav','KRo-x2uoHUg.002.wav']

    df=pd.DataFrame()

    for path in tqdm.tqdm(audio_paths):
        
        name=path.split('/')[-1]
        if (not name in blacklist) and (not name in list(results_df.filename)):

            df=get_probs(path)
            results_df=results_df.append(df)
            results_df.to_csv(out_df_path,index=False)