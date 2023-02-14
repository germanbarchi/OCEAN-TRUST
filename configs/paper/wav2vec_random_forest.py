"""
learning curve 
Iterations 100
Stratified
Features: egemaps

"""
from itertools import product
import glob 
import os,sys
import pandas as pd

# Global
exp_name=os.path.basename(__file__).split('.')[0]
results_path = os.path.join('results',exp_name)

# Data

# Labels

labels_path='data/labels/final_labels.csv'
labels_df=pd.read_csv(labels_path)

label_tags=['extraversion', 'neuroticism','agreeableness', 'conscientiousness', 'openness']

random=False

# Features

data_path = 'data/features/paper/wav2vec'
feature_list=['wav2vec.csv']

features=[os.path.join(data_path,i) for i in feature_list]

feature_df=pd.read_csv(features[0],index_col=0)

speech_ratio=False

if speech_ratio:
    feature_df=pd.merge(feature_df,labels_df[['filename','speech_ratio']],left_on='Name',right_on='filename').drop(columns='filename')

feature_tags=feature_df.columns[~feature_df.columns.isin(['Name'])]

# Subset Lists

lists_path='data/lists'
lists_=['no_music_list_manual_annot.txt']

lists=[os.path.join(lists_path,j) for j in lists_]

# Data sampling

n_samples=None

stratify=True
iterations=100

# Feature importance

feature_importance=False
top_n=10

# Bootstrapping 

n_bootstrap=0

# Modeling

seed=42

features_and_filters=[]

for i, (feat, list_) in enumerate(product(features, lists)):
    features_and_filters.append((feat,list_))

n_jobs=10
rf_n_jobs=2
