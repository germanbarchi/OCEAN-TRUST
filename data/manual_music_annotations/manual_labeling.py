import os
from playsound import playsound 


def load_list (input_list):
    with open(yamnet_music_list) as yamnet_list:
        list=yamnet_list.read().splitlines()
    return list

def start(results_path,audios_list,audio_path):

    print('------------------- Start -----------------------')
    print('* Ingrese 1 para muestras que contienen música \n* Ingrese 0 para muestras que no contienen música')
    print('-------------------------------------------------')

    with open(results_path+'/music_list.txt','w') as results_music:
        with open(results_path+'/no_music_list.txt','w') as results_no_music:
            i=1
            input_error=False
            input('ENTER para comenzar')
            for audio_sample in audios_list[:10]:                
                
                print('Progreso: %d/%d --> Audio sample:%s' %(i,len(audios_list),audio_sample.split('/')[1]))
                playsound(audio_path+audio_sample)
                
                user_input=int(input('Ingrese respuesta:'))             
                
                if user_input==1:
                    results_music.write(audio_sample+'\n')
                elif user_input==0:
                    results_no_music.write(audio_sample+'\n')
                else:
                    input_error=True                    
                    while input_error:
                        print('Valor incorrecto. Por favor, ingrese 0 o 1')
                        user_input=int(input('Ingrese respuesta:'))   
                        if user_input==0:
                            results_music.write(audio_sample+'\n')
                            input_error=False 
                        elif user_input==1:
                            results_no_music.write(audio_sample+'\n')                           
                            input_error=False                  
                i+=1                
                print('-------------------------------------------------')

if __name__=='__main__':

    yamnet_music_list='yamnet_music_0.2.txt'
    audios_list=load_list(yamnet_music_list)

    audio_path='audio/'    
 
    cwd=os.path.abspath(os.getcwd())
    results_path=os.path.join(cwd,'classification_results')

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    start(results_path,audios_list,audio_path)

