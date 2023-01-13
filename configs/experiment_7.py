"""
Cross-val 
Stratified
Iterations 100
Features: egemaps 
Restricted number of training samples: n_music_list  
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

labels_path='data/labels/final_labels.csv'
labels_df=pd.read_csv(labels_path)

label_tags=['extraversion', 'neuroticism','agreeableness', 'conscientiousness', 'openness']

# Features

data_path = 'data/features'
feature_list=['new_partitions-egemaps_all_audio.csv',
        'new_partitions-egemaps_silero_no_speech.csv',
        'new_partitions-egemaps_silero_speech.csv']

features=[os.path.join(data_path,i) for i in feature_list]

feature_df=pd.read_csv(features[0])
feature_tags=feature_df.columns[~feature_df.columns.isin(['Name','Part'])]

# Subset Lists

lists_path='data/lists'
lists_=['all_audio_complete_set.txt',
    'music_list_manual_annot.txt',
    'no_music_list_manual_annot.txt']

lists=[os.path.join(lists_path,j) for j in lists_]

# Data sampling

subset=True

with open(lists[1], 'r') as file:    # open music list 
    n_samples = len(file.readlines())

stratify=True
iterations=100

# Modeling

seed=42

features_and_filters=[]

for i, (feat, list_) in enumerate(product(features, lists)):
    features_and_filters.append((feat,list_))

n_jobs=10
rf_n_jobs=2
