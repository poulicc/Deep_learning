"""
    This code allows to generate and save the noisy versions of the signals
    signals in the correct dataset files (either in Noisy of the folders TRAIN
    VAL and TEST folders).

    This function also allows you to perform tests on :
        - the correct operation of the single_noisy_sig function in the noisy_sig.py file
        - the correct operation of the jobs once the files have been saved
    The tests are :
        - the verification of the SNR which corresponds to the desired SNR_dB
        - the verification of the noise on the spectrogram of the noisy and non-noisy signals
          in linear and logarithmic scale (decibel)
"""

import os
import sys
import glob
import re
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from noisy_sig import create_noisy_sig, single_noisy_sig

# Global variables
nb_ech = int(16e3*4) # number of samples in a signal
RSB_db = 10 # SNR
tab_noise_4s = np.load('noise_cut_4s.npy') # load the tab with all the 4s cafeteria noise
random_function='unif' # random function to choose the noise

SAVE_TRAIN=False # save the noisy signals in TRAIN/NOISY dataset
SAVE_VAL=False # save the noisy signals in VAL/NOISY dataset
SAVE_TEST=False # save the noisy signals in TEST/NOISY dataset

# Check variables
TEST_FUNCTION=False # tests on the correct operation of the single_noisy_sig function in the noisy_sig.py file
TEST_LOAD=True # tests on the correct operation of the jobs once the files have been saved

TEST_RSB=True # verification of the SNR which corresponds to the desired SNR_dB
TEST_SPECTRO=True # the verification of the noise on the spectrogram of the noisy and non-noisy signals (linear scale)
TEST_SPECTRO_DB=True # the verification of the noise on the spectrogram of the noisy and non-noisy signals (decibel scale)

#Party to save noisy files for each of the files Train, Val, TEST
if SAVE_TRAIN:
    dirpath=os.path.join("..", "..", "network", "Dataset", "Train", "Clean", "*.flac") # path to upload Train/Clean files
    savepath=os.path.join("..", "..", "network", "Dataset", "Train", "Noisy") # path to save Train/Noisy files
    data = sorted(glob.glob(dirpath)) #list of dirpath files

    for i in range (0, len(data)):
        #This line works ONLY if there are no numbers other than the file number in the path.
        #of the file in the path. CAUTION
        list = [int(s) for s in re.findall(r'-?\d+?', data[i])] # take all numbers before .flac in the path in a list
        list = ''.join(map(str, list)) #extract the number of the list

        new_sig = single_noisy_sig(data[i], RSB_db, tab_noise_4s) # create noisy signal

        path = os.path.join(savepath, list +'.flac') # path to save it
        sf.write(path, new_sig, int(16e3)) # save
        print(i/len(data)*100) # % of achievement

if SAVE_VAL:
    dirpath=os.path.join("..", "..", "network", "Dataset", "Val", "Clean", "*.flac")
    savepath=os.path.join("..", "..", "network", "Dataset", "Val", "Noisy")
    data = sorted(glob.glob(dirpath))

    for i in range (0, len(data)):
        #This line works ONLY if there are no numbers other than the file number in the path.
        #of the file in the path. CAUTION
        list = [int(s) for s in re.findall(r'-?\d+?', data[i])]
        list = ''.join(map(str, list))

        new_sig = single_noisy_sig(data[i], RSB_db, tab_noise_4s)

        path = os.path.join(savepath, list +'.flac')
        sf.write(path, new_sig, int(16e3))
        print(i/len(data)*100)

if SAVE_TEST:
    dirpath=os.path.join("..", "..", "network", "Dataset", "Test", "Clean", "*.flac")
    savepath=os.path.join("..", "..", "network", "Dataset", "Test", "Noisy")
    data = sorted(glob.glob(dirpath))

    for i in range (0, len(data)):
        #This line works ONLY if there are no numbers other than the file number in the path.
        #of the file in the path. CAUTION
        list = [int(s) for s in re.findall(r'-?\d+?', data[i])]
        list = ''.join(map(str, list))

        new_sig = single_noisy_sig(data[i], RSB_db, tab_noise_4s)

        path = os.path.join(savepath, list +'.flac')
        sf.write(path, new_sig, int(16e3))
        print(i/len(data)*100)


#TEST
#FOR ONLY ONE SIGNAL

if TEST_FUNCTION:
    namefile = os.path.join("..", "..", "network", "Dataset", "Train", "Clean", "0.flac") # path of the clean sig
    clean, _ = sf.read(namefile) # read clean
    new_signal = single_noisy_sig(namefile, RSB_db, tab_noise_4s) # create noisy sig


if TEST_LOAD:
    noisyfile = os.path.join("..", "..", "network", "Dataset", "Test", "Noisy", "2401.flac")# path of the noisy sig
    cleanfile = os.path.join("..", "..", "network", "Dataset", "Test", "Clean", "2401.flac")# path of the clean sig
    new_signal, _ = sf.read(noisyfile) # read noisy
    clean, _ = sf.read(cleanfile) # read clean


if TEST_RSB:
    Pe = np.sum((clean-new_signal)**2) # power of the 'estimation' of the clean sig
    Pc = np.sum((clean)**2)# power of the clean sig
    print("Le RSB entre les deux signaux est "+str(10*np.log10((Pc/Pe)))+'dB.') # RSB

if TEST_SPECTRO:
    N = 512 #n_fft
    H = int(N/2) #hop_length

    #stft
    noisy_stft = librosa.stft(new_signal, n_fft=N, hop_length=H)
    clean_stft  = librosa.stft(clean, n_fft=N, hop_length=H)

    #magnitude
    noisy_mag = np.abs(noisy_stft)
    clean_mag = np.abs(clean_stft)

    plt.figure()
    plt.subplot(121)
    plt.pcolormesh(noisy_mag)
    plt.colorbar()
    plt.title("Spectrogram of the new noisy signal")

    plt.subplot(122)
    plt.pcolormesh(clean_mag)
    plt.colorbar()
    plt.title("Spectrogram of the clean signal")
    plt.show()

if TEST_SPECTRO_DB:
    N = 512 #n_fft
    H = int(N/2) #hop_length

    #stft
    noisy_stft = librosa.stft(new_signal, n_fft=N, hop_length=H)
    clean_stft  = librosa.stft(clean, n_fft=N, hop_length=H)

    #magnitude
    noisy_mag = np.abs(noisy_stft)
    clean_mag = np.abs(clean_stft)

    plt.figure()
    plt.subplot(121)
    plt.pcolormesh(20*np.log10(noisy_mag))
    plt.colorbar()
    plt.title("Spectrogram of the new noisy signal")

    plt.subplot(122)
    plt.pcolormesh(20*np.log10(clean_mag))
    plt.colorbar()
    plt.title("Spectrogram of the clean signal")
    plt.show()
