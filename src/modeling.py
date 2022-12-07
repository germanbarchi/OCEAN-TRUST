from pyexpat import features
import pandas as pd
import numpy as np
import tqdm
import warnings
import glob

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.utils import resample

from joblib import Parallel, delayed
from IPython import embed

from src.utils import make_partitions

def cross_val_5_folds(df,n_train,n_val,feature_tags,label_tags,seed):
    
    metrics_list=[] 
    
    for i in range(5):
        
        df_val=df[df['partition']==i].sample(n=n_val,replace=True,random_state=seed+i)
        df_train=df[~df['partition'].isin(df_val.partition)].sample(n=n_train,replace=True,random_state=seed+i)   
     
        RF_reg= train_model (df_train,feature_tags,label_tags,seed)
        
        r2_all,MAE_all,MSE_all,RMSE_all,y_test= predict(RF_reg,df_val,feature_tags,label_tags)
        
        metrics=[r2_all,np.sqrt(r2_all),MAE_all,MSE_all,RMSE_all,i]

        metrics_list.append(metrics)
        
    metrics_list=np.transpose(metrics_list)
    df_fold=pd.DataFrame({'r2':metrics_list[0],'r':metrics_list[1],'MAE':metrics_list[2],'MSE':metrics_list[3],'RMSE':metrics_list[4],'fold':metrics_list[5]})
    
    return df_fold

def bootstrap(df,iterations,n_train,n_val,feature_tags,label_tags,seed):
    
    metrics_list=[]
    
    #df = df.dropna() 
    # train_basename_ids=df[df['part_new']=='Train']['basename'].unique()
    # val_basename_ids=df[df['part_new']=='Val'].unique()

    for i in tqdm.tqdm(range(iterations)):       
        
        # resample train partition. 
        
        #sampled_train_ids = resample(train_basename_ids, replace=True, n_samples=n_train,random_state=seed+i)
        #sampled_val_ids = resample(val_basename_ids, replace=True, n_samples=n_val,random_state=seed+i)

        # df_train=df[df['basename'].isin(sampled_train_ids)]
        # df_val=df[df['basename'].isin(sampled_val_ids)]
        
        df_train=df[df['Partition']=='Train'].sample(n=n_train,replace=True,random_state=seed+i)
        df_val=df[df['Partition']=='Val'].sample(n=n_val,replace=True,random_state=seed+i)

        RF_reg= train_model (df_train,feature_tags,label_tags,seed)
        
        r2_all,MAE_all,MSE_all,RMSE_all,y_test= predict(RF_reg,df_val,feature_tags,label_tags)
        
        metrics=[r2_all,np.sqrt(r2_all),MAE_all,MSE_all,RMSE_all]

        metrics_list.append(metrics)
        
    metrics_list=np.transpose(metrics_list)
    df=pd.DataFrame({'r2':metrics_list[0],'r':metrics_list[1],'MAE':metrics_list[2],'MSE':metrics_list[3],'RMSE':metrics_list[4]})
    df['iteration']=df.index

    return df

def bootstrap_parallel(df,iterations,n_train,n_val,feature_tags,label_tags,seed,n_jobs=1,rf_n_jobs=None):   
 
    
    def func(i):     
        
        df_train=df[df['Partition']=='Train'].sample(n=n_train,replace=True,random_state=seed+i)
        df_val=df[df['Partition']=='Val'].sample(n=n_val,replace=True,random_state=seed+i)

        RF_reg= train_model (df_train,feature_tags,label_tags,seed,rf_n_jobs=rf_n_jobs)
        
        r2_all,MAE_all,MSE_all,RMSE_all,y_test= predict(RF_reg,df_val,feature_tags,label_tags)
        
        metrics=[r2_all,np.sqrt(r2_all) if r2_all>0 else np.nan,MAE_all,MSE_all,RMSE_all]
        
        return metrics        

    metrics_list=Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in range(iterations))
    
    metrics_list=np.transpose(metrics_list)
    df=pd.DataFrame({'r2':metrics_list[0],'r':metrics_list[1],'MAE':metrics_list[2],'MSE':metrics_list[3],'RMSE':metrics_list[4]})
    df['iteration']=df.index

    return df

def bootstrap_parallel_no_partitions(df,iterations,n_train,n_val,feature_tags,label_tags,seed,n_jobs=1,rf_n_jobs=None):   
 
    def func(i):     
        
        df_train=df.sample(n=n_train,replace=True,random_state=seed+i)
        df_val=df[~df.Name.isin(df_train.Name)].sample(n=n_val,replace=True,random_state=seed+i)

        RF_reg= train_model (df_train,feature_tags,label_tags,seed,rf_n_jobs=rf_n_jobs)
        
        r2_all,MAE_all,MSE_all,RMSE_all,y_test= predict(RF_reg,df_val,feature_tags,label_tags)
        
        metrics=[r2_all,np.sqrt(r2_all) if r2_all>0 else np.nan,MAE_all,MSE_all,RMSE_all]
        
        return metrics        

    metrics_list=Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in range(iterations))
    
    metrics_list=np.transpose(metrics_list)
    df=pd.DataFrame({'r2':metrics_list[0],'r':metrics_list[1],'MAE':metrics_list[2],'MSE':metrics_list[3],'RMSE':metrics_list[4]})
    df['iteration']=df.index

    return df

def bootstrap_parallel_no_music_analysis(df,iterations,n_train,n_val,feature_tags,label_tags,seed,n_jobs=1,rf_n_jobs=None):   
 
    def func(i):     
        
        
        df_train=df.sample(n=n_train,replace=False,random_state=seed+i)
        df_val=df[~df.Name.isin(df_train.Name)].sample(n=n_val,replace=False)

        RF_reg= train_model (df_train,feature_tags,label_tags,seed,rf_n_jobs=rf_n_jobs)
        
        r2_all,MAE_all,MSE_all,RMSE_all,y_test= predict(RF_reg,df_val,feature_tags,label_tags)
        
        metrics=[r2_all,np.sqrt(r2_all) if r2_all>0 else np.nan,MAE_all,MSE_all,RMSE_all]
        
        return metrics        

    metrics_list=Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in range(iterations))
    
    metrics_list=np.transpose(metrics_list)
    df=pd.DataFrame({'r2':metrics_list[0],'r':metrics_list[1],'MAE':metrics_list[2],'MSE':metrics_list[3],'RMSE':metrics_list[4]})
    df['iteration']=df.index

    return df

def bootstrap_parallel_music_analysis(df,iterations,n_train,feature_tags,label_tags,seed,n_jobs=1,rf_n_jobs=None):   
 
    def func(i):     
        
        #Sample without replacement with n_train=900 and val=subset_size-n_train

        df_train=df.sample(n=n_train,replace=False,random_state=seed+i)
        df_val=df[~df.Name.isin(df_train.Name)]

        RF_reg= train_model (df_train,feature_tags,label_tags,seed,rf_n_jobs=rf_n_jobs)
        
        r2_all,MAE_all,MSE_all,RMSE_all,y_test= predict(RF_reg,df_val,feature_tags,label_tags)
        
        metrics=[r2_all,np.sqrt(r2_all) if r2_all>0 else np.nan,MAE_all,MSE_all,RMSE_all]
        
        return metrics        

    metrics_list=Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in range(iterations))
    
    metrics_list=np.transpose(metrics_list)
    df=pd.DataFrame({'r2':metrics_list[0],'r':metrics_list[1],'MAE':metrics_list[2],'MSE':metrics_list[3],'RMSE':metrics_list[4]})
    df['iteration']=df.index

    return df

def bootstrap_parallel_music_analysis_replacement(df,iterations,n_train,n_val,feature_tags,label_tags,seed,n_jobs=1,rf_n_jobs=None):   
    
    def func(i):     
        
        #Sample with replacement

        df_train=df.sample(n=n_train,replace=True,random_state=seed+i)
        df_val=df[~df.Name.isin(df_train.Name)].sample(n=n_val,replace=True,random_state=seed+i)

        RF_reg= train_model (df_train,feature_tags,label_tags,seed,rf_n_jobs=rf_n_jobs)
        
        r2_all,MAE_all,MSE_all,RMSE_all,y_test= predict(RF_reg,df_val,feature_tags,label_tags)
        
        metrics=[r2_all,np.sqrt(r2_all) if r2_all>0 else np.nan,MAE_all,MSE_all,RMSE_all]
        
        return metrics        
   
    metrics_list=Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in range(iterations))
    
    metrics_list=np.transpose(metrics_list)
    df=pd.DataFrame({'r2':metrics_list[0],'r':metrics_list[1],'MAE':metrics_list[2],'MSE':metrics_list[3],'RMSE':metrics_list[4]})
    df['iteration']=df.index

    return df

def bootstrap_parallel_music_analysis_replacement_split_train_var(df,iterations,n_train,n_val,feature_tags,label_tags,seed,n_jobs=1,rf_n_jobs=None):   
    
    def func(i):     
        
        #Sample without replacement with n_train=1000 and n_val=200

        df_train=df[df['Partition']=='Train'].sample(n=n_train,replace=True,random_state=seed+i)
        df_val=df[df['Partition']=='Val'].sample(n=n_val,replace=True,random_state=seed+i)

        RF_reg= train_model (df_train,feature_tags,label_tags,seed,rf_n_jobs=rf_n_jobs)
        
        r2_all,MAE_all,MSE_all,RMSE_all,y_test= predict(RF_reg,df_val,feature_tags,label_tags)
        
        metrics=[r2_all,np.sqrt(r2_all) if r2_all>0 else np.nan,MAE_all,MSE_all,RMSE_all]
        
        return metrics        
   
    metrics_list=Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in range(iterations))
    
    metrics_list=np.transpose(metrics_list)
    df=pd.DataFrame({'r2':metrics_list[0],'r':metrics_list[1],'MAE':metrics_list[2],'MSE':metrics_list[3],'RMSE':metrics_list[4]})
    df['iteration']=df.index

    return df

def cross_val_(df,iterations,n_train,n_val,feature_tags,label_tags,seed,n_jobs=1,rf_n_jobs=None):  

    def func(i):   
        metrics_list=[]
        for i in range(5):
            
            df_val=df[df['partition']==i].sample(n=n_val,replace=True,random_state=seed+i)
            df_train=df[~df['partition'].isin(df_val.partition)].sample(n=n_train,replace=True,random_state=seed+i)   
        
            RF_reg= train_model (df_train,feature_tags,label_tags,seed)
            
            r2_all,MAE_all,MSE_all,RMSE_all,y_test= predict(RF_reg,df_val,feature_tags,label_tags)
            
            metrics=[r2_all,np.sqrt(r2_all),MAE_all,MSE_all,RMSE_all,i]

            metrics_list.append(metrics)
            
        metrics_list=np.transpose(metrics_list)
        df_fold=pd.DataFrame({'r2':metrics_list[0],'r':metrics_list[1],'MAE':metrics_list[2],'MSE':metrics_list[3],'RMSE':metrics_list[4],'fold':metrics_list[5]})
        
        return df_fold

    metrics_=Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in range(iterations))
    df_final=pd.concat(metrics_)
    
    return df_final

def cross_val_speech_vs_no_speech_subset(df,feature_tags,label_tags,seed):

    n_groups=int(max(df.group))+1
    df_final=pd.DataFrame()
    for j in range(n_groups):    
        
        metrics_list=[]
        df_=df[df['group']==j]

        for i in range(5):
            
            df_val=df_[df_['partition']==i]
            df_train=df_[~df_['partition'].isin(df_val.partition)]
        
            RF_reg= train_model (df_train,feature_tags,label_tags,seed)
            
            r2_all,MAE_all,MSE_all,RMSE_all,y_test= predict(RF_reg,df_val,feature_tags,label_tags)
            
            metrics=[r2_all,np.sqrt(r2_all),MAE_all,MSE_all,RMSE_all,i]

            metrics_list.append(metrics)
            
        metrics_list=np.transpose(metrics_list)
        df_fold=pd.DataFrame({'r2':metrics_list[0],'r':metrics_list[1],'MAE':metrics_list[2],'MSE':metrics_list[3],'RMSE':metrics_list[4],'fold':metrics_list[5]})
        df_fold['group']=int(j)

        df_final=df_final.append(df_fold)

    return df_final

def bootstrap_parallel_no_partitions_learning_curve(df,iterations,feature_tags,label_tags,seed,n_jobs=1,rf_n_jobs=None):   
    
    dfs=[]
    n_samples=np.arange(1,11)*0.1

    for i in df.index:
        df.loc[i,'basename']=df.loc[i,'Name'].split('.')[0]
    
    def func(i):     

        sampled_labels=np.random.choice(df.basename.unique(),size=int(len(df.basename.unique())*step),replace=False)

        # sample 80% of sampled unique labels for train and 20% for test 

        train_labels=np.random.choice(sampled_labels,size=int(len(sampled_labels)*0.8),replace=False)
        test_labels=np.setdiff1d(sampled_labels,train_labels)

        df_train=df[df['basename'].isin(train_labels)]
        df_test=df[df['basename'].isin(test_labels)]

        RF_reg= train_model (df_train,feature_tags,label_tags,seed,rf_n_jobs=rf_n_jobs)
        
        r2_all,MAE_all,MSE_all,RMSE_all,y_test= predict(RF_reg,df_test,feature_tags,label_tags)
        
        metrics=[r2_all,np.sqrt(r2_all) if r2_all>0 else np.nan,MAE_all,MSE_all,RMSE_all]
        
        return metrics        

    for step in n_samples:
        metrics_list=Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in range(iterations))

        metrics_list=np.transpose(metrics_list)
        df_=pd.DataFrame({'r2':metrics_list[0],'r':metrics_list[1],'MAE':metrics_list[2],'MSE':metrics_list[3],'RMSE':metrics_list[4]})
        #df_['iteration']=df.index
        df_.loc[:,'%_samples']=step
        
        dfs.append(df_)    
    
    final_df=pd.concat(dfs).reset_index(drop=True)   

    return final_df
    
def train_model(df_train,feature_tags,label_tags,seed,rf_n_jobs=None):  
    
    X_train,Y_train=split_X_Y(df_train,feature_tags,label_tags)
    
    RF_reg=RandomForestRegressor(random_state=seed,n_jobs=rf_n_jobs) 
    
    RF_reg.fit(X_train.values,Y_train.values)
 
    return RF_reg

def split_X_Y(df,feature_tags,label_tags):
    
    df_X=df[feature_tags]
    df_Y=df[label_tags]

    return df_X, df_Y
  
def predict(RF_reg, val,feature_tags,label_tags):
    
    X_val, Y_val= split_X_Y(val,feature_tags,label_tags)

    predictions=RF_reg.predict(X_val.values)

    r2=r2_score(Y_val, predictions)    

    MAE=mean_absolute_error(Y_val, predictions)
    MSE=mean_squared_error(Y_val, predictions)
    RMSE=np.sqrt(mean_squared_error(Y_val, predictions))

    return r2,MAE,MSE,RMSE,Y_val,predictions


class experiments:


    def __init__(self,feature_tags,label_tags,n_folds=5,iterations=10,stratify=False,n_jobs=1,rf_n_jobs=1):
        self.feature_tags=feature_tags
        self.label_tags=label_tags
        self.n_folds=n_folds
        self.iterations=iterations
        self.stratify=stratify    
        self.rf_n_jobs=rf_n_jobs
        self.n_jobs=n_jobs
    
    def cross_val(self,df): 
        
        def func(i):
            partition=make_partitions(self.n_folds)

            feature_importance=[]
            metrics_list=[]
            predictions_all=np.array([], dtype=np.int64).reshape(0,5)
            y_val_all=pd.DataFrame()
            
            # Partitioning options

            if self.stratify: 
                df_final=partition.make_strat_folds(df) #se puede agregar random_seed
            else:
                df_final=partition.make_folds_by_id(df)

            # Run cross Val

            for fold in range(self.n_folds):
                df_val=df_final[df_final['fold']==float(fold)]
                df_train=df_final[~df_final['basename'].isin(df_val.basename)]
                RF_reg= train_model (df_train,self.feature_tags,self.label_tags,42,rf_n_jobs=self.rf_n_jobs)
                        
                r2_all,MAE_all,MSE_all,RMSE_all,y_val,predictions= predict(RF_reg,df_val,self.feature_tags,self.label_tags)
                metrics=[r2_all,np.sqrt(r2_all),MAE_all,MSE_all,RMSE_all,fold]
                metrics_list.append(metrics)
                
                predictions_all=np.concatenate((predictions_all,predictions),axis=0)
                y_val_all=pd.concat([y_val_all,y_val])

                feature_importance.append(RF_reg.feature_importances_)
            
            r2_fold=r2_score(y_val_all, predictions_all)  

            metrics_list=np.transpose(metrics_list)
            df_fold=pd.DataFrame({'r2':metrics_list[0],'r':metrics_list[1],'MAE':metrics_list[2],'MSE':metrics_list[3],'RMSE':metrics_list[4],'fold':metrics_list[5],'r2_fold':r2_fold})
            
            return df_fold

        metrics_=Parallel(n_jobs=self.n_jobs)(delayed(func)(i) for i in tqdm.tqdm(range(self.iterations)))
        df_=pd.concat(metrics_)

        return df_

def create_importance_df(importance_data,data_type,feature_tags):
    
    df_importance=pd.DataFrame()
    for i in range(len(importance_data)):
        percentil_95=np.percentile(importance_data[i],95)
        values=importance_data[i][importance_data[i]>percentil_95]
        values_indexes=np.asarray(importance_data[i]>percentil_95).nonzero()
        importance_df=pd.DataFrame({'features':feature_tags[values_indexes],'value':values,'fold':i,'data_type':data_type})
        df_importance=pd.concat([df_importance,importance_df])
    return df_importance