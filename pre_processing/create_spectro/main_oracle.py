"""
    This file allows to generate an oracle signal, that is to say a signal whose
    the modulus of the spectrogram is that of the noiseless signal and whose phase is that of a
    of a noisy spectrogram.

    This file allows to generate it but also to save it in the current folder.

    The choice of the initial signal that will be transformed into an oracle is completely arbitrary.
    It can be changed, it is currently chosen in the traning set.
"""
import sys
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from verif_function import reconstruction_sig, SNR

## Global variables : choose what you want to do during the run
SNR_=True # print the SNR of the oracle and the clean signal
PRINT_SPECTRO=True # print the spectrogram before denormalization
PRINT_SPECTRO_DB=True # print the spectrogram (in dB) before denormalization

##Definitions of data paths
sys.path.append(os.path.dirname(__file__))
CURDIRPATH=os.path.dirname(__file__)
#noisy/clean spectro path for train
TRAINNOISEPATH=os.path.join("..", "..", "network", "Dataset_spectro","Train","Noisy","*.npy")
TRAINCLEANPATH=os.path.join("..", "..", "network", "Dataset_spectro", "Train","Clean","*.npy")
#phases paths for train
TRAIN_PHASE_NOISEPATH=os.path.join("..", "..", "network", "info","Train", "X_phase_all.npy")
TRAIN_PHASE_CLEANPATH=os.path.join("..", "..", "network", "info","Train", "X_clean_phase_all.npy")
#normalization factors for train
TRAIN_NORM_PATH=os.path.join("..", "..", "network", "info","Train", "normalize_info_all.npy")

## Stock values in tab
train_noise_paths = sorted(glob.glob(TRAINNOISEPATH))#list of noisy spectrogram paths
train_clean_paths = sorted(glob.glob(TRAINCLEANPATH))#list of clean spectrogram paths
train_phase_noise=np.load(TRAIN_PHASE_NOISEPATH)#table of noisy phase
train_phase_clean=np.load(TRAIN_PHASE_CLEANPATH)#table of clean phase
train_norm=np.load(TRAIN_NORM_PATH)#table of norm info

##Start the process
num, index = 21,1020
#load spectro
noisy_spectro=np.load(train_noise_paths[num])
clean_spectro=np.load(train_clean_paths[num])
#load phase
noisy_phase=train_phase_noise[num]
clean_phase=train_phase_clean[num]
#load info norm
min,max=train_norm[num][0], train_norm[num][1]

#If you want to see the spectrogram
if PRINT_SPECTRO:
    plt.figure()
    plt.subplot(121)
    plt.pcolormesh(noisy_spectro)
    plt.colorbar()
    plt.title("Spectrogram of the new noisy signal")

    plt.subplot(122)
    plt.pcolormesh(clean_spectro)
    plt.colorbar()
    plt.title("Spectrogram of the clean signal")
    plt.show()

#If you want to see the spectrogram in dB
if PRINT_SPECTRO_DB:
    plt.figure()
    plt.subplot(121)
    plt.pcolormesh(20*np.log10(noisy_spectro))
    plt.colorbar()
    plt.title("Spectrogram of the new noisy signal")

    plt.subplot(122)
    plt.pcolormesh(20*np.log10(clean_spectro))
    plt.colorbar()
    plt.title("Spectrogram of the clean signal")
    plt.show()

#reconstruction
recons_oracle=reconstruction_sig(clean_spectro, noisy_phase, max, min,True,[True, str(index)+"_oracle.flac"])
recons_clean=reconstruction_sig(clean_spectro, clean_phase, max, min,False,[False, str(index)+"_clean.flac"])

#If you want to print the SNR
if SNR_:
    SNR_value = SNR(recons_oracle, recons_clean)
    SNR_out = SNR_value-10
    print("Le SNR est de ", SNR_value, "dB.")
    print("out ", SNR_out, "dB.")
