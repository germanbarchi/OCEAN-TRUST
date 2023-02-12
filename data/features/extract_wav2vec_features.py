import numpy as np
import librosa
from s3prl.upstream.wav2vec.hubconf import wav2vec
import torch
import s3prl
import glob
import gin
import s3prl.hub as hub
import os
import tqdm

@gin.configurable

def extract(x,device,model,layer):
        wav, fs = librosa.core.load(x, sr=16000)        
        with torch.no_grad():
            wav = torch.from_numpy(wav).unsqueeze(0).to(device)
            results = model(wav)
            features = results["hidden_states"]
        if layer is not None:
            if isinstance(layer, int):
                features = features[layer][0].detach().cpu().numpy()
            elif isinstance(layer, list):
                features = [features[i][0].detach().cpu().numpy() for i in layer]
                features = np.stack(features)
                features = np.transpose(features,(1,0,2))
        else:
            features = [li[0].detach().cpu().numpy() for li in features]
            features = np.stack(features)
            features = np.transpose(features,(1,0,2))
        
        return features

if __name__=='__main__':

    audios=glob.glob('/home/gbarchi/Documentos/Trust/Personality/OCEAN_new_structure/data/audio/*/*.wav')
    layer=None
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model= getattr(hub, 'wav2vec2')().to(device) # build the Wav2Vec 2.0 model with pre-trained weights

    path='wav2vec_features'
    if not os.path.exists(path):
        os.mkdir(path)
    
    for audio in tqdm.tqdm(audios):
        name=audio.split('/')[-1].split('.wav')[0]
        features_=extract(audios[0],device,model,layer)
        np.save(os.path.join(path,name),features_)

