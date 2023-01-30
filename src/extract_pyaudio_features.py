import sys,os 
import librosa
import glob
import opensmile
from pyAudioAnalysis import ShortTermFeatures
import argparse
import numpy as np
import pandas as pd
import tqdm
from IPython import embed

def return_names(file_paths):
    split_path=file_paths.split('/')
    file_name=split_path[-1]
    partition=split_path[-2]

    return file_name,partition
  
def pyaudio_functionals(signal,fs):

    F, f_names = ShortTermFeatures.feature_extraction(signal, fs, 0.050*fs, 0.025*fs)

    df=pd.DataFrame(F.T,columns=f_names)
    summary=df.describe()

    dict_={}
    for name in summary.columns:
        dict_feature={name+'_mean':summary.loc['mean',name].copy(),
        name+'_std':summary.loc['std',name].copy()}
        dict_.update(dict_feature)

    functionals=pd.DataFrame.from_dict([dict_])

    return functionals


def concat(df,new_row):
    
    df_functionals=pd.concat([df,new_row])

    return df_functionals

def min_max_normalization(y):
    if not len(y)==0:
        y=(y-y.min())/(y.max()-y.min())
    return y

def p95_normalization(y):
    if not len(y)==0:
        y=y/(np.percentile(y,95))
    return y

def extract_features(features_path,file_paths,duration=0,normalize=False,norm_method='p95'):
    
    FS=16000
    df=pd.DataFrame()    
        
    for file in tqdm.tqdm(file_paths):   

        file_tag, partition=return_names(file)             
        signal=librosa.core.load(file,sr=FS)[0]

        if normalize:
            if norm_method=='p95':
                signal=p95_normalization(signal)
            elif norm_method=='min_max':    
                signal=min_max_normalization(signal)
        
        if not duration==0:            
            
            n_samples=int(duration*FS)            
            if len(signal)>n_samples:
                functionals=pyaudio_functionals(signal[:n_samples],FS)
            else: 
                functionals=pd.DataFrame()
        else:
            functionals=pyaudio_functionals(signal,FS)

        if not functionals.empty:    
            functionals['Name']=file_tag 
            df=concat(df,functionals)

    df.to_csv(features_path,index=False) 

if __name__ == '__main__':
    
    argparser=argparse.ArgumentParser(description='Extract pyaudioanalysis features from dir')
    argparser.add_argument('features_path',help='Path to save features')
    argparser.add_argument('--files_path',help='Files directory')
    argparser.add_argument('--duration',help='Audio duration (seconds). Audio will be trimmed at t=duration)',default=0)
    argparser.add_argument('--normalize',help='If True, normalization will be applied. Default method is percentile 95')
    argparser.add_argument('--norm_method',help='select <p95> or <min_max>')

    args=vars(argparser.parse_args())
    
    files_path_=glob.glob(os.path.join(args['files_path'],'*/*.wav'))

    extract_features(args['features_path'],files_path_,duration=int(args['duration']),normalize=args['normalize'],norm_method=args['norm_method'])