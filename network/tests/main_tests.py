"""
    This code allows you to try the weights of files saved on unpublished audio.
NOTE that the network must be the same as the one that calculates the weights.

    It works as follows:
        1. Loading the noisy audio and its non-noisy equivalent
        2. Formatting to pass it in the linear FNN network
        3. Passing it through the network
        4. Reconstruction of the signal including a denormalization

    Two possibilities are available to us on the tests either :
        - we only want to calculate the SNR for all the audio of the TEST dataset,
          in this case, we will put RECONSTRUC_ALLSIG=True to show a curve
          SNR curve for all the data.
        - we want to see the spectrograms (linear and dB) of the input, output and clean signals
          output and clean + signal reconstructs the signal and records it.
"""
import os
import sys
import glob
import re

# Path to load the functions and the data in other folders
sys.path.append(os.path.dirname(__file__))
CURDIRPATH = os.path.dirname(__file__)
FUNCTIONPATH = os.path.join(CURDIRPATH, "..", "..", "pre_processing", "create_spectro")
sys.path.append(FUNCTIONPATH)
DATASETPATH = os.path.join(CURDIRPATH, "..", "Dataset", "Test")
sys.path.append(DATASETPATH)

import torch as th
import soundfile as sf
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import librosa
import librosa.display
from verif_function import reconstruction_sig, reconstruction_sig_dB, SNR
from spectro_norm import sig_to_spectro, sig_to_spectro_dB


## Global variables
RECONSTRUC_ONESIG=True
RECONSTRUC_ALLSIG=True
IN_DB=False

##Prepare data
#load signals noisy and clean
NOISEPATH=os.path.join(DATASETPATH,"Noisy","*.flac") #noisy spectro
CLEANPATH=os.path.join(DATASETPATH,"Clean","*.flac") #clean spectro

#sorted path
noise_paths = sorted(glob.glob(NOISEPATH))
clean_paths = sorted(glob.glob(CLEANPATH))

# model
net = nn.Sequential(
nn.Linear(257, 257),
nn.ReLU(),
nn.Linear(257, 257),
nn.ReLU(),
nn.Linear(257, 257),
)
# load weights
MODELPATH = os.path.join(CURDIRPATH, "..","model_FNN")
name_model = "test_9.pt"
model = th.load(os.path.join(MODELPATH, name_model))

if RECONSTRUC_ONESIG:
    #selec one signal and take the number of the sound
    index = 27
    list = [int(s) for s in re.findall(r'-?\d+?', noise_paths[index])] # take all numbers before .flac in the path in a list
    num = ''.join(map(str, list))

    #Read
    noisy, _= sf.read(noise_paths[index])
    clean, _ = sf.read(clean_paths[index])

    #extraction of the modulus spectrogram, the phase, and the normalization info for the noisy and the clean
    if IN_DB:
        noise_spectro, clean_spectro, phase_noise, phase_clean,normalize_info = sig_to_spectro_dB(noisy, clean)
    else:
        noise_spectro, clean_spectro, phase_noise, phase_clean,normalize_info = sig_to_spectro(noisy, clean)
    #transforms it to tensor
    noise_spectro_th = th.from_numpy(np.transpose(noise_spectro)).float()
    #output
    output_network_test1 = model(noise_spectro_th)
    #Mise en forme de np pour la suite du traitement
    output_numpy_test1 = np.transpose(output_network_test1.detach().numpy())
    #Suppression des valeurs négatives dans le spectrogramme
    output_numpy_test1[output_numpy_test1<0]=10e-8

    ##Reconstruction, print signal and save it
    if IN_DB:
        recons_noisy=reconstruction_sig_dB(output_numpy_test1, phase_noise, normalize_info[1], normalize_info[0],True,[True, str(num)+"_noisy.flac"])
    else:
        recons_noisy=reconstruction_sig(output_numpy_test1, phase_noise, normalize_info[1], normalize_info[0],True,[True, str(num)+"_noisy.flac"])

    #Spectrogram linear
    vm=0.6
    fig=plt.figure(figsize=(21, 7))
    plt.subplot(131)
    plt.pcolormesh(noise_spectro,vmax=vm)
    plt.title("Input noisy")
    plt.tight_layout()

    plt.subplot(133)
    plt.pcolormesh(clean_spectro,vmax=vm)
    plt.title("Clean")
    plt.tight_layout()
    plt.colorbar()


    plt.subplot(132)
    plt.pcolormesh(output_numpy_test1,vmax=vm)
    plt.colorbar()
    plt.title("Output")
    plt.tight_layout()
    

    #Spectrogram dB
    

    vma=0
    fig1=plt.figure(figsize=(21, 7))
    plt.subplot(131)
    plt.pcolormesh(20*np.log10(noise_spectro),vmin=-60,vmax=vma)
    plt.title("Input noisy")
    plt.tight_layout()

    plt.subplot(133)
    plt.pcolormesh(20*np.log10(clean_spectro),vmin=-60,vmax=vma)
    plt.title("Clean")
    plt.tight_layout()
    plt.colorbar()

    plt.subplot(132)
    plt.pcolormesh(20*np.log10(output_numpy_test1),vmin=-60,vmax=vma)
    plt.title("Output")
    plt.tight_layout()
    
    plt.show()
    
    

    ##SNR
    SNR_value = SNR(recons_noisy, clean)-10
    print("Le SNR est de ", SNR_value, "dB.")

if RECONSTRUC_ALLSIG:
    SNR_list=[]

    for index in range(0, len(noise_paths)):
        #Read
        noisy, _= sf.read(noise_paths[index])
        clean, _ = sf.read(clean_paths[index])
        list = [int(s) for s in re.findall(r'-?\d+?', noise_paths[index])] # take all numbers before .flac in the path in a list
        num = ''.join(map(str, list))
        #extraction of the modulus spectrogram, the phase, and the normalization info for the noisy and the clean
        if IN_DB:
            noise_spectro, clean_spectro, phase_noise, phase_clean,normalize_info = sig_to_spectro_dB(noisy, clean)
        else:
            noise_spectro, clean_spectro, phase_noise, phase_clean,normalize_info = sig_to_spectro(noisy, clean)

        noise_spectro_th = th.from_numpy(np.transpose(noise_spectro)).float()
        #output
        output_network_test1 = model(noise_spectro_th)
        #Mise en forme de np pour la suite du traitement
        output_numpy_test1 = np.transpose(output_network_test1.detach().numpy())
        #Suppression des valeurs négatives dans le spectrogramme
        output_numpy_test1[output_numpy_test1<0]=10e-8

        ##Reconstruction
        if IN_DB:
            recons_noisy=reconstruction_sig_dB(output_numpy_test1, phase_noise, normalize_info[1], normalize_info[0],False,[False, str(num)+"_noisy.flac"])
        else:
            recons_noisy=reconstruction_sig(output_numpy_test1, phase_noise, normalize_info[1], normalize_info[0],False,[False, str(num)+"_noisy.flac"])

        ##SNR
        SNR_value = SNR(recons_noisy, clean)-10
        SNR_list.append(SNR_value)

    plt.figure()
    plt.plot(SNR_list)
    plt.title("SNR for all signals in test data set")
    plt.ylabel('SNR (dB)')
    plt.xlabel('Signal number')
    plt.show()

    print(sum(SNR_list)/len(SNR_list))
    print(np.median(SNR_list))
