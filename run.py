from numpy import extract
from src.utils import format_data
from src.utils import normalize_data
from src.modeling import bootstrap_parallel_no_partitions_learning_curve,cross_val_speech_vs_no_speech_subset,cross_val, bootstrap, bootstrap_parallel, bootstrap_parallel_music_analysis, bootstrap_parallel_music_analysis_replacement, bootstrap_parallel_no_music_analysis, bootstrap_parallel_no_partitions,bootstrap_parallel_music_analysis_replacement, bootstrap_parallel_music_analysis_replacement_split_train_var
from configs.bootstrap_parallel_no_partitions_learning_curve import label_tags,feature_tags,seed,features_and_filters,labels_df, results_path, iterations#, n_train, n_val
import os
import tqdm
from pathlib import Path
from IPython import embed
import pandas as pd 

def main ():

    dfs=[]

    for i, (feat,filter) in tqdm.tqdm(enumerate(features_and_filters)):
        
        feat_df=pd.read_csv(feat)        
      
        df=format_data(feat_df,labels_df,filter)

        #df=normalize_data(df,feature_tags)

        df=bootstrap_parallel_no_partitions_learning_curve(df,iterations,feature_tags,label_tags,seed)
        
        features_name=Path(feat).stem
        filter_name=Path(filter).stem
        
        df['filter']=filter_name
        df['feature']=features_name
        dfs.append(df)
    
        df=pd.concat(dfs).reset_index(drop=True)    
        
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        df.to_csv(os.path.join(results_path,'results_speech.csv'))
            
if __name__=='__main__':

    main()

