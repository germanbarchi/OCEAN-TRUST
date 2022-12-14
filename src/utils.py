import pandas as pd
import numpy as np 
import tqdm
from IPython import embed
from tqdm.notebook import tqdm

def concat_df(DF_features,DF_labels):

    # Concat features and label dataframes

    DF=pd.merge(DF_features,DF_labels,left_on='Name',right_on='filename').drop(columns='filename')

    if 'Part' in DF.columns:
        DF=DF.drop(columns='Part')

    return DF

def filter_df(DF,filter_list):
    
    with open (filter_list,'r') as file:
        list=file.read().splitlines()
    
    filtered_df=DF[DF['Name'].isin(list)] 

    return filtered_df

def format_data(df_features, df_labels, filter_list=None):
    
    df_features=df_features.fillna(0)
    
    df=concat_df(df_features,df_labels)
    if not filter_list==None:
        df=filter_df(df,filter_list)

    return df 

def add_basename(df):

    for i in df.index:
        df.loc[i,'basename']=df.loc[i,'Name'].split('.')[0]
    
    return df

def normalize_data(df,feature_tags):

    df=add_basename(df)

    quantile_df=df.groupby('basename').quantile(.5).reset_index()
    quantile_df=pd.merge(quantile_df,df.loc[:,['Name','basename']],on='basename')

    median_df=df.groupby('basename').mean().reset_index()
    median_df=pd.merge(median_df,df.loc[:,['Name','basename']],on='basename')

    df=pd.merge(df,median_df,on='Name',suffixes=('','_(median)'))
    df=pd.merge(df,quantile_df,on='Name',suffixes=('','_(quantile)'))
    
    for i in feature_tags:
        df[i]=df[i]-df[i+'_(median)']
        df[i]=df[i]/df[i+'_(quantile)']    
    
    return df[df.columns]


class make_partitions:

    # make_folds_by_id: dataframe partition by unique ids
    # make_strat_folds: dataframe stratified partition by gender, ethnicity, labels_mean, music

    def __init__(self,n_folds):
        self.n_folds=n_folds

    def make_folds_by_id(self,df):
        
        df_out=pd.DataFrame()
        folds=[]
        
        for i in df.index:
            df.loc[i,'basename']=df.loc[i,'Name'].split('.')[0]
        
        folds_len=int(len(df.basename.unique())/5)
        unique=df.basename.unique()
        
        for i in range(self.n_folds):
            fold=np.random.choice(unique,size=folds_len,replace=False)
            folds.append(fold)
            unique =[j for j in unique if not j in fold]
            df_=df[df['basename'].isin(fold)].copy()
            df_.loc[:,'fold']=float(i)
            df_out=pd.concat([df_out,df_])
        if not len(unique)==0:
            df_rest=df[df['basename'].isin(unique)].copy()
            df_rest.loc[:,'fold']=4
            df_out=pd.concat([df_out,df_rest])
        
        return df_out

    def make_strat_folds(self,df,random_seed=None):

        df_=pd.DataFrame()        
        df=df.sample(frac=1)
        partitions=self.n_folds
        nquant = 4
        boundaries = df['labels_mean'].quantile(np.linspace(0,1,nquant))
        df['labels_mean_quant'] = np.searchsorted(boundaries, df['labels_mean'])        

        np.random.seed(random_seed)

        stratify_columns=['ethnicity','gender','music','labels_mean_quant']
        independent_columns='basename'

        
        if isinstance(partitions, int):
            partitions = [1.0 / partitions for p in range(partitions - 1)]
        elif sum(partitions) >= 1:
            raise Exception('Partitions proportions must sum less than 1')
        partitions.append(1 - sum(partitions))

        n_sets = len(partitions)

        partidx = {i: [] for i in range(n_sets)}

        if independent_columns is not None:
            groups = df.groupby(stratify_columns).groups.items()
            dist = {k: len(v) * np.array(partitions) for k, v in groups}
            part = {k: np.zeros(n_sets) for k, v in groups}

            for k, g in tqdm(df.groupby(independent_columns)):
                diffs = []
                for i, r in g.iterrows():
                    group = tuple(r[stratify_columns].to_list())
                    diffs.append(dist[group] - part[group])
                pix = np.max(np.array(diffs), 0).argmax()
                gix = np.max(np.array(diffs), 1).argmax()
                partidx[pix].extend(g.index)
                group = tuple(g.iloc[gix][stratify_columns].to_list())
                part[group][pix] += len(g)

        else:
            for k, g in df.groupby(stratify_columns):
                ix = np.tile(np.arange(n_sets),len(g)//n_sets)
                ix = np.hstack([ix,np.random.choice(np.arange(n_sets),len(g)-len(ix), replace=False)])
    #             ix = np.random.permutation(
    #                 sum([[i] * int(np.ceil(len(g) * p))
    #                      for i, p in enumerate(partitions)], []))[:len(g)]
                for i in range(n_sets):
                    partidx[i].extend(g[ix == i].index)        

        pix = partidx

        for k,v in pix.items():
            df.loc[v,'fold'] = int(k)
            #df.loc[v,'group'] = int(i)
        df_=df_.append(df)

        return df_