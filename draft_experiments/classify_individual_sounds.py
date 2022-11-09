from IPython import embed
from pathlib import Path
import sys
from importlib.machinery import SourceFileLoader
import numpy as np
import glob
import pandas as pd

main_path=Path.joinpath(Path(__name__).resolve().parents[1])
parent_dir=str(Path.joinpath(main_path,'src/toolbox.py'))
toolbox = SourceFileLoader("yamnet_classifier", parent_dir).load_module()

model_dir= str(Path.joinpath(main_path,'yamnet/models/yamnet.h5'))
map_file= str(Path.joinpath(main_path,'yamnet/models/research/audioset/yamnet/yamnet_class_map.csv'))

data_path='/home/gbarchi/Documentos/Trust/Personality/OCEAN_new_structure/data/audio/'

dict_={}

for i in glob.glob(data_path+'*/*'):
    
    filename=i.split('/')[-1]
    pred,classes=toolbox.yamnet_classifier(i,model_dir,map_file)

    music_pred=pred[list(classes).index('Music')]

    dict_['filename']=music_pred

a=pd.DataFrame.from_dict(dict_)
a.to_csv('music_predictions_test.csv')    
