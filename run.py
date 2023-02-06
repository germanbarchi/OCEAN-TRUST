import os,sys,tqdm,json,importlib
from pathlib import Path
#from IPython import embed
import pandas as pd
import argparse

from src.utils import format_data
from src.utils import normalize_data
from src.modeling import experiments

sys.path.append('./configs')

def main (exp_dict):

    for experiment,method in zip (exp_dict.keys(),exp_dict.values()):
        print(experiment)
        print(method)
        configs=importlib.import_module(experiment)

        exp=experiments(configs.feature_tags,configs.label_tags,n_folds=5,iterations=configs.iterations,stratify=configs.stratify,rf_n_jobs=configs.rf_n_jobs,n_jobs=configs.n_jobs,n_samples=configs.n_samples,seed=configs.seed,n_bootstrap=configs.n_bootstrap,random=configs.random) 
        
        dfs=[]
        dfs_boot=[]

        for i, (feat,filter) in tqdm.tqdm(enumerate(configs.features_and_filters)):
            
            if not os.path.exists(configs.results_path):
                os.makedirs(configs.results_path)

            feat_df=pd.read_csv(feat)        
        
            df=format_data(feat_df,configs.labels_df,filter)

            #df=normalize_data(df,feature_tags)

            df,df_boot=exp.__getattribute__(method)(df)
            
            features_name=Path(feat).stem
            filter_name=Path(filter).stem
            
            df['filter']=filter_name
            df['feature']=features_name
            dfs.append(df)
            df_out=pd.concat(dfs).reset_index(drop=True)  

            if len(configs.label_tags)==5:
                trait='All'
            else:
                trait=configs.label_tags[0]

            if not configs.n_bootstrap==0: 
                df_boot['filter']=filter_name
                df_boot['feature']=features_name
                dfs_boot.append(df_boot)
                dfs_boot_out=pd.concat(dfs_boot).reset_index(drop=True)         
                
                dfs_boot_out.loc[:,'trait']=trait
                dfs_boot_out.to_csv(os.path.join(configs.results_path,'results_bootstrapping.csv'))

            df_out.loc[:,'trait']=trait            
            df_out.to_csv(os.path.join(configs.results_path,'results.csv'))
            
if __name__=='__main__':

    argparser=argparse.ArgumentParser(description='Run experiments')
    argparser.add_argument('--multiple',help='True to run multiple experiments',default=False)
    args=vars(argparser.parse_args())

    if args['multiple']:
        config_file='experiments.JSON'
    else:
        config_file='individual_experiments.JSON'
        
    with open (config_file) as jsonfile:
        experiment_dict=json.load(jsonfile)

    main(experiment_dict)

