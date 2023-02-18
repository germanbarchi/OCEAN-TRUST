import os
from playsound import playsound 
import glob
import pandas as pd 
from IPython import embed
from scipy.io import wavfile
import numpy as np

def start(results_path,audio_path):
    
    if os.path.exists(results_path): 
        df_results=pd.read_csv(results_path)
    else:
        df_results=pd.DataFrame({'name':[],'music':[]})

    audios_list=glob.glob(audio_path+'/*/*.wav')

    print('------------------- Start -----------------------')
    print('* Press 1 for audio samples containing music \n* Press 0 for audio samples not containing music \n* Press any key to repeat audio')
    print('-------------------------------------------------')
    
    i=1
    input_error=False
    input('Press ENTER to start')
    for k, audio_sample in enumerate(audios_list):                
        filename=audio_sample.split('/')[-1]                
        
        if not df_results['name'].eq(filename).any():
            print('Progreso: %d/%d --> Audio sample:%s' %(i,len(audios_list),filename))
            fs, data = wavfile.read(audio_sample)                    
            
            #sample 1s at the begining, 1s at middle and 1s at the end of file

            trimmed_data=np.concatenate([data[:fs],data[7*fs:8*fs],data[14*fs:]])
            wavfile.write('aux.wav', fs, trimmed_data)
            playsound('aux.wav')
            
            user_input=input('Enter value:')             
            
            if int(user_input)==1:
                df_results.loc[k,'music']=1
                df_results.loc[k,'name']=filename
            elif int(user_input)==0:
                df_results.loc[k,'music']=0
                df_results.loc[k,'name']=filename
            else:                        
                input_error=True                    
                while input_error:
                    print('Listen again and enter value, 0 or 1')                            
                    playsound('aux.wav')
                    user_input=int(input('Enter value:'))   
                    if user_input==0:
                        df_results.loc[k,'music']=0
                        df_results.loc[k,'name']=filename
                        input_error=False 
                    elif user_input==1:
                        df_results.loc[k,'music']=1
                        df_results.loc[k,'name']=filename                           
                        input_error=False                  
            i+=1
            df_results.to_csv('df_music_labels_all.csv')                
            print('-------------------------------------------------')

if __name__=='__main__':

    audio_path='/home/german/Documentos/Trust/First_Impressions_Dataset/OCEAN_new_structure/data/audio'    
    results_path='df_music_labels_all.csv'

    start(results_path,audio_path)

