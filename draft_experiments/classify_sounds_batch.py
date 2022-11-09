from IPython import embed
from pathlib import Path
import sys
from importlib.machinery import SourceFileLoader
import numpy as np 

main_path=Path.joinpath(Path(__name__).resolve().parents[1])
parent_dir=str(Path.joinpath(main_path,'src/toolbox.py'))
toolbox = SourceFileLoader("yamnet_classifier", parent_dir).load_module()

model_dir= str(Path.joinpath(main_path,'yamnet/models/yamnet.h5'))
map_file= str(Path.joinpath(main_path,'yamnet/models/research/audioset/yamnet/yamnet_class_map.csv'))
#data_path=str(Path.joinpath(main_path,'/   '))
data_save_path=str(Path.joinpath(main_path,'data/silero/no_speech_classification'))

toolbox.yamnet_classifier_batch(data_path,data_save_path,model_dir,map_file)

preds=np.load('temp/predictions.npy')
print(preds)