import sys 
import librosa
import glob
import opensmile

import pandas as pd
import tqdm


def smile(signal,fs):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        sampling_rate=None,
        resample=False)
    df_functionals=smile.process_signal(signal,fs)
    
    return df_functionals

def concat(df,new_row):
    
    df_functionals=pd.concat([df,new_row])

    return df_functionals

def extract_features(features_path,file_paths):

    FS=16000
    df=pd.DataFrame()    
         
    for file in tqdm.tqdm(file_paths):   

        file_tag=file.split('/')[-1]
                
        signal=librosa.core.load(file,sr=FS)[0]

        functionals=smile(signal,FS)
            
        functionals['Name']=file_tag

        if df.empty:        
            df=functionals
        else:
            df=concat(df,functionals)

    df.to_csv(features_path,index=False)


if __name__=='__main__':
    
    features_path='/home/gbarchi/Documentos/Trust/OCEAN-TRUST/data/features/new_partitions-egemaps_silero_no_speech_th_0.1.csv'
    data_path=glob.glob('/home/gbarchi/Documentos/Trust/OCEAN-TRUST/data/silero/no_speech_0.1/*.wav')

    extract_features(features_path,data_path)