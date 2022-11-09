import sys 
import librosa
import glob
import opensmile
import os
import pathlib
import pandas as pd
import tqdm
from IPython import embed

def return_names(file_paths):
    split_path=file_paths.split('/')
    file_name=split_path[-2]

    return file_name

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

def extract_features(features_save_path,file_paths):

    FS=16000
    df=pd.DataFrame()    
         
    for file in tqdm.tqdm(file_paths):   

        filename=return_names(file)
                
        signal=librosa.core.load(file,sr=FS)[0]

        functionals=smile(signal,FS)
            
        functionals['Name']=filename+'.wav'

        if df.empty:        
            df=functionals
        else:
            df=concat(df,functionals)

    df.to_csv(features_save_path,index=False)


if __name__=='__main__':
    main_path=os.getcwd()
    parents=pathlib.Path(main_path).parents[0]
    
    all=glob.glob(os.path.join(parents,'data/source_separation/separated_audios/mdx_extra_q/*/*'))
    no_voice=glob.glob(os.path.join(parents,'data/source_separation/separated_audios/mdx_extra_q/*/no*'))

    voice=[]
    for i in all:
        if not 'no_' in i:
            voice.append(i)

    data_dict={'voice':voice,'no_voice':no_voice}
    
    for i in data_dict.keys():
    
        extract_features(str(pathlib.Path.joinpath(parents,'data/features/music_analysis_'+i+'_egemaps_all_audio.csv')),data_dict[i])