"""
    This file is a verification file of the previously formed elements.
    It serves no purpose in the process other than to check that what has been done so far
    done so far works.

    Thus, in this code is tested:
        - the storage of spectrograms for the FNN network (NETWORK_TYPE="FNN")
        - the storage of spectrograms for the CNN network (NETWORK_TYPE="CNN")
        - storage of other elements in the info folder
        - the reconstruction of the signals
        - the initial SNR
    for TRAIN (VERIF_TRAIN=True) and VALIDATION (VERIF_VAL=True) files
"""
import sys
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from verif_function import reconstruction_sig, SNR

# Global variables to choose what you want to do
NETWORK_TYPE="FNN" # or CNN
VERIF_TRAIN = True
VERIF_VAL = False
SNR_=True # print the SNR
RECONSTRUC=True # reconstruction of signalss
PRINT_INFO=True # print dimensions of tabs, spectrograms, min max normalization

#Definitions of data paths
sys.path.append(os.path.dirname(__file__))
CURDIRPATH=os.path.dirname(__file__)

if NETWORK_TYPE=="CNN":
    TRAINNOISEPATH=os.path.join("..", "..", "network", "Dataset_spectro","Train","Noisy","*.npy") #noisy spectro train
    TRAINCLEANPATH=os.path.join("..", "..", "network", "Dataset_spectro", "Train","Clean","*.npy") #clean spectro train

    VALNOISEPATH=os.path.join("..", "..", "network", "Dataset_spectro","Val","Noisy","*.npy")#noisy spectro val
    VALCLEANPATH=os.path.join("..", "..", "network", "Dataset_spectro", "Val","Clean","*.npy")#clean spectro val

if NETWORK_TYPE=="FNN":
    TRAIN_X_NOISEPATH=os.path.join("..", "..", "network", "Dataset", "Train","Noisy","X_all.npy")#noisy spectro train FILE
    TRAIN_X_CLEANPATH=os.path.join("..", "..", "network", "Dataset", "Train","Clean","X_clean_all.npy")#clean spectro train FILE

    VAL_X_NOISEPATH=os.path.join("..", "..", "network", "Dataset", "Val","Noisy","X_all.npy")#noisy spectro val FILE
    VAL_X_CLEANPATH=os.path.join("..", "..", "network", "Dataset", "Val","Clean","X_clean_all.npy")#clean spectro val FILE

TRAIN_PHASE_NOISEPATH=os.path.join("..", "..", "network", "info","Train", "X_phase_all.npy") #train phase noise
TRAIN_PHASE_CLEANPATH=os.path.join("..", "..", "network", "info","Train", "X_clean_phase_all.npy")#train phase clean

VAL_PHASE_NOISEPATH=os.path.join("..", "..", "network", "info","Val", "X_phase_all.npy")#val phase noise
VAL_PHASE_CLEANPATH=os.path.join("..", "..", "network", "info","Val", "X_clean_phase_all.npy")#val phase clean

TRAIN_NORM_PATH=os.path.join("..", "..", "network", "info","Train", "normalize_info_all.npy") #norm info train
VAL_NORM_PATH=os.path.join("..", "..", "network", "info","Val", "normalize_info_all.npy") #norm info val

#Stock variables in tabs
train_phase_noise=np.load(TRAIN_PHASE_NOISEPATH)
train_phase_clean=np.load(TRAIN_PHASE_CLEANPATH)

val_phase_noise=np.load(VAL_PHASE_NOISEPATH)
val_phase_clean=np.load(VAL_PHASE_CLEANPATH)

train_norm=np.load(TRAIN_NORM_PATH)
val_norm=np.load(VAL_NORM_PATH)

#Definitions of the list of the paths
if NETWORK_TYPE=="CNN":
    train_noise_paths = sorted(glob.glob(TRAINNOISEPATH))
    train_clean_paths = sorted(glob.glob(TRAINCLEANPATH))

    val_noise_paths = sorted(glob.glob(VALNOISEPATH))
    val_clean_paths =sorted(glob.glob(VALCLEANPATH))

#Load files
if NETWORK_TYPE=="FNN":
    train_noise = np.load(TRAIN_X_NOISEPATH)
    train_clean = np.load(TRAIN_X_CLEANPATH)

    val_noise = np.load(VAL_X_NOISEPATH)
    val_clean =np.load(VAL_X_CLEANPATH)

if VERIF_TRAIN:
    num, index =0,0 #its a choice, to know the numbers check index files

    #load spectro
    if NETWORK_TYPE=="CNN":
        noisy_spectro=np.load(train_noise_paths[num])
        clean_spectro=np.load(train_clean_paths[num])
    if NETWORK_TYPE=="FNN":
        noisy_spectro=np.transpose(train_noise[num*251:(num+1)*251])
        clean_spectro=np.transpose(train_clean[num*251:(num+1)*251])

    #load phase
    noisy_phase=train_phase_noise[num]
    clean_phase=train_phase_clean[num]

    #load la info norm
    min,max=train_norm[num][0], train_norm[num][1]


    if PRINT_INFO:
        PRINT_SPECTRO=True
        PRINT_SPECTRO_DB=False
        print("Les dimensions du spectro noisy/clean sont ",noisy_spectro.shape, "/", clean_spectro.shape)
        print("Les dimensions des phases noisy/clean sont ",noisy_phase.shape, "/", clean_phase.shape)
        print("min, max", min,max)

        print("\nLes spectrogrammes bruts sont les suivants")

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

    if RECONSTRUC:
        recons_noisy=reconstruction_sig(noisy_spectro, noisy_phase, max, min,True,[True, str(index)+"_noisy.flac"])
        recons_clean=reconstruction_sig(clean_spectro, clean_phase, max, min,True,[True, str(index)+"_clean.flac"])

        if SNR_:
            SNR_value = SNR(recons_noisy, recons_clean)
            print("Le SNR est de ", SNR_value, "dB.")

if VERIF_VAL:
    num, index = 43, 1658#voir le fichier txt
    #load un spectro (pas le 1er) de train
    if NETWORK_TYPE=="CNN":
        noisy_spectro=np.load(val_noise_paths[num])
        clean_spectro=np.load(val_clean_paths[num])
    if NETWORK_TYPE=="FNN":
        noisy_spectro=np.transpose(val_noise[num*251:(num+1)*251])
        clean_spectro=np.transpose(val_clean[num*251:(num+1)*251])
    #load la phase
    noisy_phase=val_phase_noise[num]
    clean_phase=val_phase_clean[num]
    #load la infonorm
    min,max=val_norm[num][0], val_norm[num][1]


    if PRINT_INFO:
        PRINT_SPECTRO=False
        PRINT_SPECTRO_DB=False
        print("Les dimensions du spectro noisy/clean sont ",noisy_spectro.shape, "/", clean_spectro.shape)
        print("Les dimensions des phases noisy/clean sont ",noisy_phase.shape, "/", clean_phase.shape)
        print("min, max", min,max)

        print("\nLes spectrogrammes bruts sont les suivants")

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

    if RECONSTRUC:
        recons_noisy=reconstruction_sig(noisy_spectro, noisy_phase, max, min,True,[True, str(index)+"_noisy.flac"])
        recons_clean=reconstruction_sig(clean_spectro, clean_phase, max, min,True,[True, str(index)+"_clean.flac"])

        if SNR_:
            SNR_value = SNR(recons_noisy, recons_clean)
            print("Le SNR est de ", SNR_value, "dB.")
