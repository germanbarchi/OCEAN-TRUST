import glob
import json
import pandas as pd
import os
import tqdm
from pathlib import Path

def main(labels,base_path,out_path):

    dict_={'Train':{},'Test':{},'Val':{}}

    for split in ['Train', 'Test', 'Val']:

        dict_[split]['labels']={'openness':0,'conscientiousness':1,'extraversion':2,'agreeableness':3,'neuroticism':4}

        metadata=[]
        for i,r in tqdm.tqdm(labels.iterrows()):
            if Path(base_path,split,r['filename']).exists():
                ind_dict = {'path':str(Path(split,r['filename'])),
                        'labels':[r['openness'],r['conscientiousness'],r['extraversion'],r['agreeableness'],r['neuroticism']]
                        }
                metadata.append(ind_dict)

        dict_[split]['metadata']=metadata

        with open(Path(out_path,f'{split}_meta_data.json'), "w") as outfile:
            json.dump(dict_[split], outfile)

if __name__=='__main__':
    
    labels=pd.read_csv('/home/gbarchi/Documentos/Trust/OCEAN-TRUST/data/labels/final_labels.csv')
    base_path='/home/gbarchi/Documentos/Trust/Personality/OCEAN_new_structure/data/audio'
    out_path="."
    
    main(labels,base_path,out_path)