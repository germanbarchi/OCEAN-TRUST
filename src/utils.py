import pandas as pd

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