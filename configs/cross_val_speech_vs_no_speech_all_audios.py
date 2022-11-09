"""
Experiment description
"""
from itertools import product
import glob 
import os
import pandas as pd

# Global

results_path = "results/cross_val-speech-vs-no_speech_subset"

# Data

# Labels

labels_path='data/labels/first_impressions_parts-20_groups-5_folds.csv'
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

lists=['data/lists/all_audio_complete_set.txt']

# Data sampling

#n_train=4000
#n_val=800    
#iterations=100

# Modeling

seed=42

features_and_filters=[]

for i, (feat, list_) in enumerate(product(features, lists)):
    features_and_filters.append((feat,list_))

bootstrapping_n_jobs=10
rf_n_jobs=2
