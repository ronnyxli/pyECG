
# data I/O
import os
import wfdb
import scipy.io as sio
import pickle

# quantitative
import numpy as np

# signal processing
from scipy import signal, fftpack

from compress_1D import compress, reconstruct
from sig_proc import proc_ecg
from calc_ecg_features import calc_rr

# plotting
from matplotlib import pyplot as plt

# debugging
import pdb


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

    # grab record f1o01 from https://physionet.org/physiobank/database/fantasia
    # record = wfdb.rdsamp('f1o02', pb_dir='fantasia')
    record = pickle.load( open('data/f1o01.pkl', 'rb') )

    # get sampling rate and save signals in dict
    fs = record[1]['fs']
    data = {}
    for n in range(0, len(record[1]['sig_name'])):
        data[record[1]['sig_name'][n]] = record[0][:,n]

    # extract 10-second chunk
    ecg = data['ECG'][0:fs*14]
    resp = data['RESP'][0:fs*14]

    # pre-processing
    ecg_proc = proc_ecg(ecg, fs)

    # sig_features = analysis(sig, data['fs'])

    # extract R-R intervals and heart rate from ECG signal
    # RR = calc_rr(ecg_proc)

    # derive heart rate from R-R intervals
    # HRV = calc_hrv(RR)

    # derive respiration waveform and breathing rate from R-R peak amplitudes
    # resp = calc_resp(RR)

    # compression
    compressed_data = compress(ecg_proc, {'energyThresh':0.9, 'quantPrecision': 8})
    
    # TODO: reconstruction

    pdb.set_trace()
