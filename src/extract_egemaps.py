import sys,os 
import librosa
import glob
import opensmile
import argparse
import numpy as np
import pandas as pd
import tqdm
from IPython import embed
import random

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

def extract_features(features_path,file_paths,duration=0,random_sampling=False,normalize=False,norm_method='p95',blacklist_dir=''):
    
    FS=16000
    df=pd.DataFrame()    
    
    if not blacklist_dir:
        with open (blacklist,'r') as blist:
            blacklist=blist.read().splitlines()
    else:
        blacklist=[]

    for file in tqdm.tqdm(file_paths):   

        if not file.split('/')[-1] in blacklist:

            file_tag, partition=return_names(file)             
            signal=librosa.core.load(file,sr=FS)[0]

            if normalize:
                if norm_method=='p95':
                    signal=p95_normalization(signal)
                elif norm_method=='min_max':    
                    signal=min_max_normalization(signal)
            
            if not duration==0:       
                n_samples=int(duration*FS)

                if random_sampling:
                    max_start_index=len(signal)-n_samples
                    start=random.randint(0,max_start_index)
                    end=start+n_samples
                else:
                    start=0
                    end=n_samples

                if len(signal)>(duration):
                    functionals=smile(signal[start:end],FS)
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
    argparser.add_argument('--random_sampling',help='extract random audio fragments of length <duration>',default=False)
    argparser.add_argument('--normalize',help='If True, normalization will be applied. Default method is percentile 95')
    argparser.add_argument('--norm_method',help='select <p95> or <min_max>')
    argparser.add_argument('--blacklist',help='blacklist text files containing samples names to be removed')
    args=vars(argparser.parse_args())
    
    files_path_=glob.glob(os.path.join(args['files_path'],'*/*.wav'))

    extract_features(args['features_path'],files_path_,duration=int(args['duration']),random_sampling=args['random_sampling'],normalize=args['normalize'],norm_method=args['norm_method'],blacklist_dir=args['blacklist'])