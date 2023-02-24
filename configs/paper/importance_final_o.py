"""
Cross-val 
Stratified
Iterations 100
Bootstrapping in test partition: 10 iterations
Features: all speech_ratio+egemaps groups combinations  
audio_input: 
    * full-audio
    * speech
Filters: manual annotations no music 

"""
from itertools import product
import glob 
import os,sys
import pandas as pd

# Global
exp_name=os.path.basename(__file__).split('.')[0]
results_path = os.path.join('results/paper',exp_name)

# Model

model='random_forest'

# Labels

labels_path='data/labels/final_labels+speech_ratio.csv'
labels_df=pd.read_csv(labels_path)

label_tags=['openness']

random=False

# Features

data_path = 'data/features/paper/egemaps'
feature_list=['egemaps_full_audio.csv',
        'egemaps_speech.csv']

features=[os.path.join(data_path,i) for i in feature_list]

# feature tag format {<tag>:[<feature_label>]}

feature_tags = {'sr':['speech_ratio'],'frequency' : [
            'F0semitoneFrom27.5Hz_sma3nz_amean',
            'F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
            'F0semitoneFrom27.5Hz_sma3nz_percentile20.0',
            'F0semitoneFrom27.5Hz_sma3nz_percentile50.0',
            'F0semitoneFrom27.5Hz_sma3nz_percentile80.0',
            'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2',
            'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope',
            'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope',
            'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope',
            'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope',
            'jitterLocal_sma3nz_amean',
            'jitterLocal_sma3nz_stddevNorm',
            'F1frequency_sma3nz_amean',
            'F1frequency_sma3nz_stddevNorm',
            'F1bandwidth_sma3nz_amean',
            'F1bandwidth_sma3nz_stddevNorm',
            'F2frequency_sma3nz_amean',
            'F2frequency_sma3nz_stddevNorm',
            'F2bandwidth_sma3nz_amean',
            'F2bandwidth_sma3nz_stddevNorm',
            'F3frequency_sma3nz_amean',
            'F3frequency_sma3nz_stddevNorm',
            'F3bandwidth_sma3nz_amean',
            'F3bandwidth_sma3nz_stddevNorm'
            ],
        
        'energy' : [
            'shimmerLocaldB_sma3nz_amean',
            'shimmerLocaldB_sma3nz_stddevNorm',
            'loudness_sma3_amean',
            'loudness_sma3_stddevNorm',
            'loudness_sma3_percentile20.0',
            'loudness_sma3_percentile50.0',
            'loudness_sma3_percentile80.0',
            'loudness_sma3_pctlrange0-2',
            'loudness_sma3_meanRisingSlope',
            'loudness_sma3_stddevRisingSlope',
            'loudness_sma3_meanFallingSlope',
            'loudness_sma3_stddevFallingSlope',
            'HNRdBACF_sma3nz_amean',
            'HNRdBACF_sma3nz_stddevNorm',
            'equivalentSoundLevel_dBp'
            ],
        
        'spectral' : [
            'F1amplitudeLogRelF0_sma3nz_amean',
            'F1amplitudeLogRelF0_sma3nz_stddevNorm',
            'F2amplitudeLogRelF0_sma3nz_amean',
            'F2amplitudeLogRelF0_sma3nz_stddevNorm',
            'F3amplitudeLogRelF0_sma3nz_amean',
            'F3amplitudeLogRelF0_sma3nz_stddevNorm',
            'logRelF0-H1-H2_sma3nz_amean',
            'logRelF0-H1-H2_sma3nz_stddevNorm',
            'logRelF0-H1-A3_sma3nz_amean',
            'logRelF0-H1-A3_sma3nz_stddevNorm',
            'spectralFlux_sma3_amean',
            'spectralFlux_sma3_stddevNorm',
            'mfcc1_sma3_amean',
            'mfcc1_sma3_stddevNorm',
            'mfcc2_sma3_amean',
            'mfcc2_sma3_stddevNorm',
            'mfcc3_sma3_amean',
            'mfcc3_sma3_stddevNorm',
            'mfcc4_sma3_amean',
            'mfcc4_sma3_stddevNorm'
            ],
        
        'spectral_voiced' : [
            'alphaRatioV_sma3nz_amean',
            'alphaRatioV_sma3nz_stddevNorm',
            'hammarbergIndexV_sma3nz_amean',
            'hammarbergIndexV_sma3nz_stddevNorm',
            'slopeV0-500_sma3nz_amean',
            'slopeV0-500_sma3nz_stddevNorm',
            'slopeV500-1500_sma3nz_amean',
            'slopeV500-1500_sma3nz_stddevNorm',
            'mfcc1V_sma3nz_amean',
            'mfcc1V_sma3nz_stddevNorm',
            'mfcc2V_sma3nz_amean',
            'mfcc2V_sma3nz_stddevNorm',
            'mfcc3V_sma3nz_amean',
            'mfcc3V_sma3nz_stddevNorm',
            'mfcc4V_sma3nz_amean',
            'mfcc4V_sma3nz_stddevNorm',
            'spectralFluxV_sma3nz_amean',
            'spectralFluxV_sma3nz_stddevNorm'
            ],
        
        'unvoiced' : [
            'alphaRatioUV_sma3nz_amean',
            'hammarbergIndexUV_sma3nz_amean',
            'slopeUV0-500_sma3nz_amean',
            'slopeUV500-1500_sma3nz_amean',
            'spectralFluxUV_sma3nz_amean'
            ],
        
        'temporal' : [
            'VoicedSegmentsPerSec',
            'MeanUnvoicedSegmentLength',
            'StddevUnvoicedSegmentLength',
            'loudnessPeaksPerSec',
            'MeanVoicedSegmentLengthSec',
            'StddevVoicedSegmentLengthSec'
            ]}

multi_feature_eval=True # Computes all combinations between groups and type of features,
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

n_bootstrap=0

# Modeling

seed=42

features_and_filters=[]

for i, (feat, list_) in enumerate(product(features, lists)):
    features_and_filters.append((feat,list_))

n_jobs=10
rf_n_jobs=2
