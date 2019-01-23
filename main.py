
# data I/O
import os
import wfdb
import scipy.io as sio
import pickle

# quantitative
import numpy as np

# signal processing
from scipy import signal, fftpack

# plotting
from matplotlib import pyplot as plt

from compress_1D import compress, reconstruct

# debugging
import pdb


def get_data(db_name):
    '''
    Pulls data from the PhysioNet database specified by db_name
    '''

    # TODO: loop through all records in db and create list of data_dicts

    data_dict = {}
    pdb.set_trace()
    record = wfdb.rdsamp('100', pb_dir=db_name)

    # record is a tuple containing a numpy ndarray (the signal) and a dict (signal descriptors)

    # load relevant descriptors into dict
    data_dict['fs'] = record[1]['fs']
    data_dict['schema'] = record[1]['sig_name']

    # derive time vector in sec
    data_dict['t'] = np.linspace(0,record[1]['sig_len']/record[1]['fs'],num=record[1]['sig_len'])

    # load signals into dict based on schema
    sig = record[0]
    for n in range(0,len(data_dict['schema'])):
        data_dict[data_dict['schema'][n]] = sig[:,n]

    return data_dict


def analysis(y,fs):
    '''
    Time-series and frequency analysis of input signal
        Args: y = input signal, fs = sampling frequency
        Out: Time and frequency domain features of signal y
    '''

    out = {}

    y = y[0:fs*5-1] # extract 5-second segment

    # calculate PDF using 4-second windows with 50% overlap
    f,Pyy = signal.welch(y, fs=fs, nperseg=2*fs, noverlap=fs)

    plt.plot(f, Pyy)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.show()
    plt.close()

    out['max_freq'] = f[np.argmax(Pyy)] # dominant frequency

    # TODO: calculate energy ratio in 5-15 Hz and 5-40 Hz bands

    # histogram distribution
    y_counts, y_bins = np.histogram(y)

    return out







if __name__ == "__main__":

    # Combined measurement of ECG, breathing and seismocardiogram (CEBS database)

    # grab data from https://physionet.org/physiobank/database/cebsdb
    # print('Loading signals from CEBS database...')
    # data = get_data('mitdb')
    data = pickle.load(open('data/data.pkl', 'rb'))

    ecg = data['I'][0:10000]
    fs = data['fs']

    compress(ecg)

    pdb.set_trace()


    # plot measured ECG
    f, ax = plt.subplots(2, sharex=True)
    ax[0].plot(data['t'], data['I'])
    ax[0].plot(data['t'], data['II'])


    # sig_features = analysis(sig, data['fs'])

    # extract R-R intervals and heart rate from raw
    RR = calc_rr(sig, data['fs'])

    # derive heart rate from R-R intervals
    HRV = calc_hrv(RR)

    # derive respiration waveform and breathing rate from R-R peak amplitudes
    resp = calc_resp(RR)

    # TODO: compression/decompression

    pdb.set_trace()
