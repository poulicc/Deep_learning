"""

        Here are fonctions to normalize signal form libraSpeech

        This file contains fonctions :
            -cut_sig :
                For a whole signal from LibraSpeech suppress the beginning
                and the end with no voice (amplitude < maxi)
            -tab_sig :
                Creat a tensor of signal of length T seconds from all signals
                in dataset libraSpeech

"""

import os
import torchaudio
import torch
import soundfile as sf

def cut_sig(sig, maxi):
    """
        For a whole signal from LibraSpeech suppress the beginning
        and the end with no voice (amplitude < maxi)

        Inputs
            - sig : list containing the samples of the signal
            - maxi : threshold considered to cut the signal

        Outputs
            - signal cut
    """

    #Cut the begining with no info
    i=0
    while(i<sig.shape[0] and sig[i]/maxi<0.1):
        i+=1

    #Cut the end with no info
    j=sig.shape[0]-1
    while(j>0 and sig[j]/maxi<0.1):
        j-=1

    return(sig[i:j])


def tab_sig(T,fech=16000):

    """
        Creat a tensor of signal of length T seconds from all signals in dataset libraSpeech
        And save the new signal (cut) in the right folder according to the id of the speaker

        Inputs
            - T : lenght of the signal in second
            - fech : sampling frequency

        Outputs
            - tensor of signal of length T seconds
    """

    #Bibliothèque
    data = torchaudio.datasets.LIBRISPEECH(root='.', url = 'dev-clean')
    #Charge
    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    #Données
    dataset=data_loader.dataset
    #number of sample for a 4s signal
    n_ech=int(T*fech)
    #Nombre de signal dans le dataset
    n_dataset=len(dataset)
    #Tab final
    sample = torch.tensor(())

    #number of the signal in the dataset
    j_sig_data=0

    maxi=max(dataset[0][0][0])

    while(j_sig_data<n_dataset):
        #on prend le j-ieme signal du dataset
        print(j_sig_data/n_dataset*100, '%')
        sig=cut_sig(dataset[j_sig_data][0][0],maxi)

        #if the signal is long enough we take it in the normalize dataset

        #ARBITRORY CHOICES
        #If the id of the speaker is in [7850,7976,8297, 8842] => the signal belongs to the test dataset
        #If the id of the speaker is in [5536, 5694, 5895, 6241, 6295, 6313, 6319, 6345] => the signal belongs to the val dataset
        #For any other id, it belongs to the training set.
        if(sig.shape[0]>n_ech):
            sample=torch.cat((sample,sig[0:n_ech].reshape(n_ech,1)),1)
            sig=sig[0:n_ech].reshape(n_ech,1)

            #normalisation
            sig = sig/torch.max(sig)

            if dataset[j_sig_data][3] in [7850,7976,8297, 8842] :
                #save in Test dataset
                path = os.path.join("..", "..", "network", "Dataset", "Test", "Clean", str(j_sig_data))
                sf.write(path+'.flac', sig, int(16e3))

            elif dataset[j_sig_data][3] in [5536, 5694, 5895, 6241, 6295, 6313, 6319, 6345] :
                #save in Val dataset
                path = os.path.join("..", "..", "network", "Dataset", "Val", "Clean", str(j_sig_data))
                sf.write(path+'.flac', sig, int(16e3))
            else:
                #save in Train dataset
                path = os.path.join("..", "..", "network", "Dataset", "Train", "Clean", str(j_sig_data))
                sf.write(path+'.flac', sig, int(16e3))

        else:
            pass
        j_sig_data+=1


    return sample
