"""
Experiment description
"""
from itertools import product
import glob 
import os
import pandas as pd

# Global

results_path = "results/bootstrapping_t1000_v200_no_partitions"

# Data

# Labels

labels_path='data/labels'
labels_file=glob.glob(os.path.join(labels_path,'*'))[0]

labels_df=pd.read_csv(labels_file)
label_tags=labels_df.columns[~labels_df.columns.isin(['audio_tag','Partition'])]

# Features

data_path = 'data/features'
features=glob.glob(os.path.join(data_path,'new_partitions*'))

feature_df=pd.read_csv(features[0])
feature_tags=feature_df.columns[~feature_df.columns.isin(['Name','Part'])]

# Subset Lists

lists_path='data/lists'
lists=glob.glob(os.path.join(lists_path,'*'))

# Data sampling

n_train=1000
n_val=200    
iterations=1000

# Modeling

seed=42

features_and_filters=[]

for i, (feat, list_) in enumerate(product(features, lists)):
    features_and_filters.append((feat,list_))

bootstrapping_n_jobs=10
rf_n_jobs=2
