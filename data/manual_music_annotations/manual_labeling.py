import os
from playsound import playsound 
import glob
import pandas as pd 

def start(results_path,audio_path):
    
    df_results=pd.DataFrame()
    audios_list=glob.glob(audio_path+'/*.wav')

    print('------------------- Start -----------------------')
    print('* Ingrese 1 para muestras que contienen música \n* Ingrese 0 para muestras que no contienen música')
    print('-------------------------------------------------')
    
    with open(results_path+'/music_list.txt','w') as results_music:
        with open(results_path+'/no_music_list.txt','w') as results_no_music:
            i=1
            input_error=False
            input('ENTER para comenzar')
            for k, audio_sample in enumerate(audios_list):                
                filename=audio_sample.split('/')[-1]
                print('Progreso: %d/%d --> Audio sample:%s' %(i,len(audios_list),filename))
                playsound(audio_sample)
                
                user_input=int(input('Ingrese respuesta:'))             
                
                if user_input==1:
                    results_music.write(filename+'\n')
                    df_results.loc[k,'music']=1
                    df_results.loc[k,'name']=filename
                elif user_input==0:
                    results_no_music.write(filename+'\n')
                    df_results.loc[k,'music']=0
                    df_results.loc[k,'name']=filename
                else:
                    input_error=True                    
                    while input_error:
                        print('Valor incorrecto. Por favor, ingrese 0 o 1')
                        user_input=int(input('Ingrese respuesta:'))   
                        if user_input==0:
                            results_no_music.write(filename+'\n')
                            df_results.loc[k,'music']=0
                            df_results.loc[k,'name']=filename
                            input_error=False 
                        elif user_input==1:
                            results_music.write(filename+'\n')
                            df_results.loc[k,'music']=1
                            df_results.loc[k,'name']=filename                           
                            input_error=False                  
                i+=1
                df_results.to_csv('df_music_labels.csv')                
                print('-------------------------------------------------')

if __name__=='__main__':

    audio_path='audio/'    
 
    results_path='classification_results'

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    start(results_path,audio_path)

