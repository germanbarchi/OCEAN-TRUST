import subprocess
import os
import pathlib
from IPython import embed
from importlib.machinery import SourceFileLoader
from pathlib import Path

parent_dir=pathlib.Path(os.getcwd())

import sys

#sys.path.append('../src')
#from src.toolbox import silero_timestamps

def yamnet_classifier_batch(data_path, data_save_path,model_path,map):
    
    code_path=pathlib.Path.joinpath(parent_dir.parent,'yamnet/models/research/audioset/yamnet/inference_batch.py')
    
    subprocess.run(['python',str(code_path), data_path, data_save_path, model_path, map])


def yamnet_classifier(data_path):

    model_dir= str(Path.joinpath(parent_dir.parent,'yamnet/models/research/audioset/yamnet/yamnet.h5'))
    map_file= str(Path.joinpath(parent_dir.parent,'yamnet/models/research/audioset/yamnet/yamnet_class_map.csv'))
    code_path=str(pathlib.Path.joinpath(parent_dir.parent,'yamnet/models/research/audioset/yamnet/modules/inference.py'))    
    yamnet = SourceFileLoader("classify", code_path).load_module()    
    prediction,yamnet_classes=yamnet.classify(data_path, model_dir, map_file)

    return prediction,yamnet_classes

def silero_timestamps(file,dinamic_threshold):
 
    timestamps, speech_segments, non_speech_segments, sr=silero_timestamps(file, dinamic_threshold)

    return timestamps, speech_segments,non_speech_segments, sr
