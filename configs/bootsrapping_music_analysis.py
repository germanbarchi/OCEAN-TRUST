"""
Experiment description
"""
from itertools import product,combinations
import glob 
import os
import pandas as pd
import pathlib 

# Global
main_path=pathlib.Path(__name__).parent.absolute()
results_path = "results/bootstrapping_music_analysis"

# Data


# Labels

labels_path='data/labels'
labels_file=glob.glob(os.path.join(labels_path,'*'))[0]

labels_df=pd.read_csv(labels_file)
label_tags=labels_df.columns[~labels_df.columns.isin(['audio_tag','Partition'])]

# Features

data_path = 'data/features'
features=glob.glob(os.path.join(data_path,'music_*'))
features.append(os.path.join(data_path,'new_partitions-egemaps_all_audio.csv'))

feature_df=pd.read_csv(features[0])
feature_tags=feature_df.columns[~feature_df.columns.isin(['Name','Part'])]

# Subset Lists

lists_path=os.path.join(main_path,'data/lists')
lists=os.path.join(lists_path,'yamnet_music_0.2.txt')

# Data sampling

n_train=900

iterations=1000

# Modeling

seed=42

features_and_filters=[]

features_and_filters=[(feat,lists) for feat in features]

bootstrapping_n_jobs=10
rf_n_jobs=2



