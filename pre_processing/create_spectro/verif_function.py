"""
    In this file are functions that allow, from a spectrogram,
    to reconstruct the complete signal with the help of normalization coefficients, and the
    phases of the signals, stored in files.

    This file contains the following functions:
        - denormalization_spectro : denormalize spectrogram module
        - reconstruction_sig : reconstruct a signal from a module and a phase
        - SNR : Compute the output gain of the neural network for one output validation signal

    !!! These functions are also found in post processing !!!
"""

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa

def denormalization_spectro(spectro_mod,denorm_maxi,denorm_mini):
    """
        Denormalize spectrogram module

        Inputs
            - spectro_mod : tensor of the spectrogram module
            - denorm_maxi, denorm_mini : information needed to denormalize the module

        Outputs
            - tensor of the spectrogram module denormalized
    """

    spectro_mod = spectro_mod*(denorm_maxi-denorm_mini) + denorm_mini

    return spectro_mod

def reconstruction_sig(spectro_mod,spectro_ang,denorm_maxi,denorm_mini, plot=False, save=[False, "filename"]):
    """
        Reconstruct a signal from a module and a phase.
        Inputs
            - spectro : tensor of the noisy signal spectrogram (imaginary and real part)
            [a sup] plot : bool representation of the signal after reconstruction
            [a sup] save : bool save the signal in .flac format

        Output
            - sig : tensor of format 'torch.Size([number of value])'.
    """
    N = 512
    H = int(N/2)


    mod_denorm=denormalization_spectro(spectro_mod,denorm_maxi,denorm_mini)

    expo_angle = np.exp(1j*spectro_ang)

    reconstruc = mod_denorm * expo_angle

    sig = librosa.istft(reconstruc, n_fft=N, hop_length=int(H))

    if plot==True:
        plt.figure()
        plt.plot(sig)
        plt.title("Le signal reconstitué")
        plt.show()

    if save[0]==True:
        sf.write(save[1], sig, int(16e3))

    return sig

def reconstruction_sig_dB(spectro_mod,spectro_ang,denorm_maxi,denorm_mini, plot=False, save=[False, "filename"]):
    """
        Reconstruct a signal from a module and a phase.
        Inputs
            - spectro : tensor of the noisy signal spectrogram (imaginary and real part)
            [a sup] plot : bool representation of the signal after reconstruction
            [a sup] save : bool save the signal in .flac format

        Output
            - sig : tensor of format 'torch.Size([number of value])'.
    """
    N = 512
    H = int(N/2)


    mod_denorm=denormalization_spectro(spectro_mod,denorm_maxi,denorm_mini)

    expo_angle = np.exp(1j*spectro_ang)

    mod_denorm = 10**(mod_denorm/20)

    reconstruc = mod_denorm * expo_angle

    sig = librosa.istft(reconstruc, n_fft=N, hop_length=int(H))

    if plot==True:
        plt.figure()
        plt.plot(sig)
        plt.title("Le signal reconstitué")
        plt.show()

    if save[0]==True:
        sf.write(save[1], sig, int(16e3))

    return sig

def SNR(out_signal, clean_signal):
    """
    Compute the output gain of the neural network for one output validation signal
        Inputs
            - out_signal : output signal from the neural network
            - clean_signal : signal without noise
            - in_SNR : SNR of input signals in the net

        Output
            - Gain : output gain of the network for one validation signal
    """
    e_hat=out_signal-clean_signal
    Pe_hat=np.sum(e_hat**2)
    Ps = np.sum(clean_signal**2)
    SDR_out=10*np.log10(Ps/Pe_hat)

    return SDR_out
