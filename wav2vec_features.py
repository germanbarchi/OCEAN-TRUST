import glob
import numpy as np
import pandas as pd 
from joblib import Parallel,delayed
import tqdm

def process(path):
    df_final=pd.read_csv('wav2vec.csv')
    df=pd.DataFrame({})
    data=np.load(path)
    name=paths[0].split('/')[-1].replace('npy','wav')
    for frame in range(data.shape[0]): 
        array=data[frame].flatten()
        df=pd.concat([df,pd.DataFrame(array.reshape(1,-1))])
        df.loc[0,'Name']=name    
    df_final=pd.concat([df_final,df])
    df_final.to_csv('wav2vec.csv')


if __name__=='__main__':

    features='/home/gbarchi/Documentos/Trust/OCEAN-TRUST/data/features/paper/wav2vec_features'
    paths=glob.glob(features+'/*.npy') 
    df_final=pd.DataFrame()
    Parallel(n_jobs=-1)(delayed(process(path)) for path in tqdm.tqdm(paths))
    