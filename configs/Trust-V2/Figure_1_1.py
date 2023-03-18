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

labels_path='data/labels/TrustV2_labels.csv'
labels_df=pd.read_csv(labels_path)

label_tags=['extraversion', 'neuroticism','agreeableness', 'conscientiousness', 'openness']

random=False

# Features

data_path = 'data/features/Trust-V2_dataset'
feature_list=['TrustV2_free-egemaps+sr-full_audio.csv',
        'TrustV2_lecture-egemaps+sr-full_audio.csv']

features=[os.path.join(data_path,i) for i in feature_list]

feature_df=pd.read_csv(features[0])

feature_tags_=feature_df.columns[~feature_df.columns.isin(['Name','Part','start','end'])]

feature_tags={'egemaps':feature_tags_}

multi_feature_eval=False # Computes all combinations between groups and type of features,
individual_features=False # Will train individual models with 1 features as input

# Subset Lists

lists_path='data/lists'
lists_=['no_music_list_manual_annot.txt']

lists=[os.path.join(lists_path,j) for j in lists_]

# Data sampling

  
n_samples=None # number of samples to create subset or 'None' to use all data 

stratify=True
iterations=10

# Feature importance 

feature_importance=False
top_n=10

# Bootstrapping 

n_bootstrap=100

# Modeling

seed=42

features_and_filters=[]

for i, (feat, list_) in enumerate(product(features, lists)):
    features_and_filters.append((feat,list_))

n_jobs=10
rf_n_jobs=2
