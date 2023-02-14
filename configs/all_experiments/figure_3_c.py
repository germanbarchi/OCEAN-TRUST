"""
Cross-val (multiple durations)
Stratified
Iterations 100
Features: egemaps
bootstrap=10
music manual annotations 

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

labels_path='data/labels/labels_no_sr.csv'
labels_df=pd.read_csv(labels_path)

label_tags=['extraversion', 'neuroticism','agreeableness', 'conscientiousness', 'openness']

random=False

# Features

data_path = 'data/features/features_duration_sr_fixed'
feature_list=['egemaps_all_audio_duration_1_random.csv',
            'egemaps_all_audio_duration_3_random.csv',
            'egemaps_all_audio_duration_5_random.csv',
            'egemaps_all_audio_duration_7_random.csv',
            'egemaps_all_audio_duration_9_random.csv',
            'egemaps_all_audio_duration_11_random.csv',
            'egemaps_all_audio_duration_13_random.csv',
            'egemaps_all_audio_duration_15_random.csv']

features=[os.path.join(data_path,i) for i in feature_list]

feature_df=pd.read_csv(features[0])

speech_ratio=False

feature_tags=feature_df.columns[~feature_df.columns.isin(['Name','Part','start','end'])]

# Subset Lists

lists_path='data/lists'
lists_=['no_music_list_manual_annot.txt']

lists=[os.path.join(lists_path,j) for j in lists_]

# Data sampling

n_samples=None 
  
stratify=True
iterations=100

n_bootstrap=10

# Modeling

seed=42

features_and_filters=[]

for i, (feat, list_) in enumerate(product(features, lists)):
    features_and_filters.append((feat,list_))

n_jobs=10
rf_n_jobs=2
