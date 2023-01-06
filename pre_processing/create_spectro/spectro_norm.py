"""
This module spectro_norm contains the functions which allow to make the
the spectrogram (with their normalization):
        - normalization: normalization of a spectrogram
        - sig_to_spectro: production of spectrograms (and phases) of the noise
             and its associated non-noisy file
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa

def normalization(mag_spectro_noisy,mag_spectro_clean):
    """
        This function does the minmax normalization between 0 and 1 of the
        spectrograms of the signal in input.

        Input :
            - mag_spectro : stft magnitude of a signal
                            -- type 'np.ndarray'

        Return :
            - mag_spectro_norm : normalized stft magnitud of a signal (same size as the input)
                                 -- type 'numpy.ndarray'
    """

    #Set new minimum and miximum arbitrarily
    new_min = 0.
    new_max = 1.

    #Conversion tensor to numpy to anticipate the rest of the processing
    min_spectro = np.min(mag_spectro_noisy)
    max_spectro = np.max(mag_spectro_noisy)

    mag_spectro_norm_noisy = (mag_spectro_noisy-min_spectro)/(max_spectro-min_spectro) * new_max + new_min
    mag_spectro_norm_clean = (mag_spectro_clean-min_spectro)/(max_spectro-min_spectro) * new_max + new_min

    return mag_spectro_norm_noisy, mag_spectro_norm_clean, min_spectro, max_spectro


def sig_to_spectro(noise, clean, plot_noise=False, plot_noise_norm=False):
    """
        This function calculates the stft of the noisy and non-noisy versions of
        the same signal in order to extract the spectrogram and the imaginary of the
        signal in its two versions.

        Input :
            - noise : noisy signal
                      -- type 'numpy.ndarray'
            - clean : clean version of the same signal
                      -- type 'numpy.ndarray'

            - plot_noise [optional]:  plot the spectrogram of the noisy signal
                                      -- type boolean

            - plot_noise_norm [optional]: plot the spectrogram of the normalized
                                          noisy signal
                                          -- type boolean

        Return :
            - noise_mag : normalized stft magnitude of the noisy signal
                          -- type 'numpy.ndarray'
            - clean_mag : normalized stft magnitude of the clean signal
                          -- type 'numpy.ndarray'
            - phase_noise : phase of the noisy signal
                            -- type 'numpy.ndarray'
            - phase_clean : phase of the clean signal
                            -- type 'numpy.ndarray'

    """
    N = 512 #n_fft
    H = int(N/2) #hop_length


    #stft
    noise_stft = librosa.stft(noise, n_fft=N, hop_length=H)
    clean_stft = librosa.stft(clean, n_fft=N, hop_length=H)

    #phases and magnitudes
    phase_noise = np.angle(noise_stft)
    phase_clean = np.angle(clean_stft)
    noise_mag = np.abs(noise_stft)
    clean_mag = np.abs(clean_stft)

    if plot_noise:
        plt.figure()
        plt.pcolormesh(noise_mag)
        plt.colorbar()
        plt.title("Spectrogram of the new noisy signal")
        plt.show()

    noise, clean, mini, maxi = normalization(noise_mag,clean_mag)
    normalize_info=[mini,maxi]

    if plot_noise_norm:
        plt.figure()
        plt.pcolormesh(noise)
        plt.colorbar()
        plt.title("Spectrogram of the new normalized noisy signal")
        plt.show()

    return noise, clean, phase_noise, phase_clean, normalize_info

def sig_to_spectro_dB(noise, clean, plot_noise=False, plot_noise_norm=False):
    """
        This function calculates the stft of the noisy and non-noisy versions of
        the same signal in order to extract the spectrogram and the imaginary of the
        signal in its two versions. This time in dB and not in linear.

        Input :
            - noise : noisy signal
                      -- type 'numpy.ndarray'
            - clean : clean version of the same signal
                      -- type 'numpy.ndarray'

            - plot_noise [optional]:  plot the spectrogram of the noisy signal
                                      -- type boolean

            - plot_noise_norm [optional]: plot the spectrogram of the normalized
                                          noisy signal
                                          -- type boolean

        Return :
            - noise_mag : normalized stft magnitude of the noisy signal
                          -- type 'numpy.ndarray'
            - clean_mag : normalized stft magnitude of the clean signal
                          -- type 'numpy.ndarray'
            - phase_noise : phase of the noisy signal
                            -- type 'numpy.ndarray'
            - phase_clean : phase of the clean signal
                            -- type 'numpy.ndarray'

    """
    N = 512 #n_fft
    H = int(N/2) #hop_length


    #stft
    noise_stft = librosa.stft(noise, n_fft=N, hop_length=H)
    clean_stft = librosa.stft(clean, n_fft=N, hop_length=H)

    #phases and magnitudes
    phase_noise = np.angle(noise_stft)
    phase_clean = np.angle(clean_stft)

    abs_noise = np.abs(noise_stft)
    abs_clean = np.abs(clean_stft)

    abs_noise[abs_noise==0]=10**(-8)
    abs_clean[abs_clean==0]=10**(-8)

    noise_mag = 20*np.log10(abs_noise)
    clean_mag = 20*np.log10(abs_clean)

    if plot_noise:
        plt.figure()
        plt.pcolormesh(noise_mag)
        plt.colorbar()
        plt.title("Spectrogram of the new noisy signal")
        plt.show()

    noise, clean, mini, maxi = normalization(noise_mag,clean_mag)
    normalize_info=[mini,maxi]

    if plot_noise_norm:
        plt.figure()
        plt.pcolormesh(noise)
        plt.colorbar()
        plt.title("Spectrogram of the new normalized noisy signal")
        plt.show()

    return noise, clean, phase_noise, phase_clean, normalize_info
