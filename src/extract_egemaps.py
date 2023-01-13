import sys,os 
import librosa
import glob
import opensmile
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
                functionals=smile(signal[:n_samples],FS)
            else: 
                functionals=pd.DataFrame()
        else:
            functionals=smile(signal,FS)

        if not functionals.empty:    
            functionals['Part']=partition
            functionals['Name']=file_tag 
            df=concat(df,functionals)

    df.to_csv(features_path,index=False) 

if __name__ == '__main__':
    
    argparser=argparse.ArgumentParser(description='Extract egemaps from dir')
    argparser.add_argument('features_path',help='Path to save features')
    argparser.add_argument('--files_path',help='Files directory')
    argparser.add_argument('--duration',help='Audio duration (seconds). Audio will be trimmed at t=duration)',default=0)
    argparser.add_argument('--normalize',help='If True, normalization will be applied. Default method is percentile 95')
    argparser.add_argument('--norm_method',help='select <p95> or <min_max>')

    args=vars(argparser.parse_args())
    
    files_path_=glob.glob(os.path.join(args['files_path'],'*/*.wav'))

    extract_features(args['features_path'],files_path_,duration=int(args['duration']),normalize=args['normalize'],norm_method=args['norm_method'])