"""
This module column_spectro contains the function which allow to make the
tables of the spectrogram columns:
        - fill_tab_column: creation of the table of columns of the spectrograms
             for all the data for the noisy signals and the clean signals
"""
import numpy as np

def fill_tab_column(nb_samples, noisy_tab, clean_tab):
    """
        This function creates the data tables that will be provided to the
        neural network. Each of the tables contains all the columns of the
        spectrograms of all the available signals (given by nb samples).

        Input :
        nb_samples, noisy_tab, clean_tab, save_tab
            - nb_samples : number of signals
                           -- int
            - noisy_tab : normalized stft magnitude of the noisy signal
                          -- type 'numpy.ndarray'

            - clean_tab : normalized stft magnitude of the clean signal
                          -- type 'numpy.ndarray'

            - save_tab [optional]: saves the noisy signal at adress save_tab[1]
                 and saves the clean signal at adress save_tab[1]

        Return :
        X, X_clean
            - X : normalized stft magnitude of all noisy signals
                  -- type 'numpy.ndarray'
            - X_clean : normalized stft magnitude of all clean signals
                        -- type 'numpy.ndarray'

    """
    #initialization of the arrays containing the columns of all the stfts
    X_clean = np.zeros((nb_samples*noisy_tab.shape[2], noisy_tab.shape[1]))
    X = np.zeros((nb_samples*noisy_tab.shape[2], noisy_tab.shape[1]))
    #extraction of columns from stfts and association in the new data table
    for i in range(0, nb_samples):
        for j in range(0, noisy_tab.shape[2]):
            extract = noisy_tab[i, :, j]
            extract_clean = clean_tab[i, :, j]
            X[i*noisy_tab.shape[2]+j, :] = extract
            X_clean[i*noisy_tab.shape[2]+j, :] = extract_clean

    return X, X_clean
