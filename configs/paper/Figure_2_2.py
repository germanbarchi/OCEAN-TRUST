"""
Cross-val 
Stratified
Iterations 100
Bootstrapping in test partition: 10 iterations
Features: egemaps+sr
audio_input: 
    * full-audio
    * speech
Filters: manual annotations no music 

"""
from itertools import product
import glob 
import os,sys
import pandas as pd
from IPython import embed

# Global
exp_name=os.path.basename(__file__).split('.')[0]
results_path = os.path.join('results/paper',exp_name)

# Model

model='random_forest'

# Labels

labels_path='data/labels/final_labels+speech_ratio.csv'
labels_df=pd.read_csv(labels_path)

label_tags=['extraversion', 'neuroticism','agreeableness', 'conscientiousness', 'openness']

random=False

# Features

data_path = 'data/features/paper/egemaps'
feature_list=['egemaps_full_audio.csv',
        'egemaps_speech.csv']

features=[os.path.join(data_path,i) for i in feature_list]

feature_df=pd.read_csv(features[0])

speech_ratio=True

if speech_ratio:
    feature_df=pd.merge(feature_df,labels_df[['filename','speech_ratio']],left_on='Name',right_on='filename').drop(columns='filename')

feature_tags=feature_df.columns[~feature_df.columns.isin(['Name','Part','start','end'])]

multi_feature_eval=False # Computes all combinations between groups and type of features,
individual_features=False # Will train individual models with 1 features as input

# Subset Lists

lists_path='data/lists'
lists_=['no_music_list_manual_annot.txt']

lists=[os.path.join(lists_path,j) for j in lists_]

# Data sampling

n_samples=None # number of samples to create subset or 'None' to use all data 

stratify=True
iterations=100

# Feature importance 

feature_importance=False
top_n=10

# Bootstrapping 

n_bootstrap=10

# Modeling

seed=42

features_and_filters=[]

for i, (feat, list_) in enumerate(product(features, lists)):
    features_and_filters.append((feat,list_))

n_jobs=10
rf_n_jobs=2
