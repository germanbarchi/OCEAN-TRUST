"""
* Training: Cross-val 
* Splits: Divided by video_ID and stratified
* Seeds: 10
* Bootstrapping in test partition: 100 iterations
* Features: egemaps
* Audio inputs: 
    * full wav
    * speech-only
* Subset: "manual_annotations_no_music" 

"""
from itertools import product
import glob 
import os,sys
import pandas as pd

# Global
exp_name=os.path.basename(__file__).split('.')[0]
results_path = os.path.join('results/paper_results/data',exp_name)

# Model

model='random_forest'

# Labels

labels_path='data/labels/final_labels.csv'
labels_df=pd.read_csv(labels_path)

label_tags=['extraversion', 'neuroticism','agreeableness', 'conscientiousness', 'openness']

random=False

# Features

data_path = 'data/features/paper/features_duration_sr'
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

if speech_ratio:
    feature_df=pd.merge(feature_df,labels_df[['filename','speech_ratio']],left_on='Name',right_on='filename').drop(columns='filename')

feature_labels=feature_df.columns[~feature_df.columns.isin(['Name','Part','start','end'])]

feature_tags={'egemaps':feature_labels}

multi_feature_eval=False # Computes all combinations between groups and type of features,
individual_features=False # Will train individual models with 1 features as input

# Subset Lists

lists_path='data/lists'
lists_=['no_music_list_manual_annot.txt']

lists=[os.path.join(lists_path,j) for j in lists_]

# Data sampling

n_samples=None 
  
stratify=True
iterations=10

n_bootstrap=100

# Feature importance 

feature_importance=False
top_n=10

# Modeling

seed=42

features_and_filters=[]

for i, (feat, list_) in enumerate(product(features, lists)):
    features_and_filters.append((feat,list_))

n_jobs=10
rf_n_jobs=2
