import pandas as pd
import numpy as np 

def concat_df(DF_features,DF_labels):

    # Concat features and label dataframes

    DF=pd.merge(DF_features,DF_labels,left_on='Name',right_on='audio_tag').drop(columns='audio_tag')

    if 'Part' in DF.columns:
        DF=DF.drop(columns='Part')

    return DF

def filter_df(DF,filter_list):
    
    with open (filter_list,'r') as file:
        list=file.read().splitlines()
    
    filtered_df=DF[DF['Name'].isin(list)] 

    return filtered_df

def format_data(df_features, df_labels, filter_list=None):
    
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

def make_partitions(df,
                    partitions,
                    stratify_columns,
                    independent_columns=None,
                    random_seed=None):

    np.random.seed(random_seed)

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
    return partidx

def make_parts(n_groups,df):
    
    df_=pd.DataFrame()
    
    for i in range(n_groups):
        
        df=df.sample(frac=1)
        
        nquant = 4
        boundaries = df['labels_mean'].quantile(np.linspace(0,1,nquant))
        df['labels_mean_quant'] = np.searchsorted(boundaries, df['labels_mean'])

        pix = make_partitions(df,5, ['ethnicity','gender','music','labels_mean_quant'], independent_columns='basename')

        for k,v in pix.items():
            df.loc[v,'partition'] = int(k)
            df.loc[v,'group'] = int(i)
        df_=df_.append(df)
    
    return df_