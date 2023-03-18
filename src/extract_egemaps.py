import sys,os
from turtle import back 
import librosa
import glob
import opensmile
import argparse
import numpy as np
import pandas as pd
import tqdm
from IPython import embed
import random

sys.path.append('silero_VAD')

import silero

from joblib import Parallel,delayed

def bool_eval(flag):
    if flag=='True':
        out=True
    else:
        out=False    
    return out 

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


def check_data(features_path,blacklist_dir):
    
    if os.path.exists(features_path):
        df=pd.read_csv(features_path)
        ommit_samples=list(df['Name'].values)
    else:
        df=pd.DataFrame()
        ommit_samples=[]

    if blacklist_dir:
        with open (blacklist_dir,'r') as blist:
            blacklist=blist.read().splitlines()
    else:
        blacklist=[]
    
    return df,ommit_samples,blacklist

def extract_features(file_path,features_path,duration=0,random_sampling=False,normalize=False,norm_method='p95',speech_ratio_option=False,ommit_samples=[],blacklist=''):
    
    FS=16000  
    print(file_path)

    name=file_path.split('/')[-1]
    
    if (not name in blacklist) and (not name in ommit_samples):

        file_tag, partition=return_names(file_path)             
        signal=librosa.core.load(file_path,sr=FS)[0]

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
                end=start+n_samples-1
            else:
                start=0
                end=n_samples-1

            if (len(signal)>(duration)):
                functionals=smile(signal[start:end],FS)
            else: 
                functionals=pd.DataFrame()
        else:
            functionals=smile(signal,FS)
            start=0
            end=int(duration*FS)-1

        if not functionals.empty:    
            functionals['Part']=partition
            functionals['Name']=file_tag 
            functionals['path']=file_path
            if speech_ratio_option:
                speech_ratio= silero.silero_timestamps(file_path,start,end,0.5)[4]
                functionals['speech_ratio']=speech_ratio

        if not os.path.isfile(features_path):
            functionals.to_csv(features_path)
        else: 
            functionals.to_csv(features_path, mode='a', header=False)

if __name__ == '__main__':
    
    argparser=argparse.ArgumentParser(description='Extract egemaps from dir')
    argparser.add_argument('features_path',help='Path to save features')
    argparser.add_argument('--files_path',help='Files directory')
    argparser.add_argument('--paths_list',help='List of files')
    argparser.add_argument('--duration',help='Audio duration (seconds). Audio will be trimmed at t=duration)',default=0)
    argparser.add_argument('--random_sampling',help='extract random audio fragments of length <duration>',default=False)
    argparser.add_argument('--normalize',help='If True, normalization will be applied. Default method is percentile 95',default=False)
    argparser.add_argument('--norm_method',help='select <p95> or <min_max>',default='p95')
    argparser.add_argument('--blacklist',help='blacklist text files containing samples names to be removed',default=[])
    argparser.add_argument('--speech_ratio',help='enables speech ratio computing',default=False)
    argparser.add_argument('--n_jobs',help='multiplocessing threads',default=1)
    args=vars(argparser.parse_args())

    if args['files_path']:       
        files_path_=glob.glob(os.path.join(args['files_path'],'*/*.wav'))
    elif args['paths_list']:
        with open (args['paths_list'],'r') as file:
            files_path_=file.read().splitlines()

    df,ommit,b_list=check_data(args['features_path'],blacklist_dir=args['blacklist'])    

    Parallel(n_jobs=int(args['n_jobs']))(delayed(extract_features)(
    file_path,args['features_path'],duration=int(args['duration']),random_sampling=bool_eval(args['random_sampling']),normalize=bool_eval(args['normalize']),
    norm_method=args['norm_method'],speech_ratio_option=bool_eval(args['speech_ratio']),ommit_samples=ommit,blacklist=b_list) for file_path in tqdm.tqdm(files_path_))
