
import os
import wfdb
import scipy.io as sio

import numpy as np
from scipy import signal, fftpack
# import sig_proc

from matplotlib import pyplot as plt

import pdb


def get_data(db_name):
    '''
    Pulls data from the PhysioNet database specified by db_name
    '''

    # TODO: loop through all records in db and create list of data_dicts

    data_dict = {}

    record = wfdb.rdsamp('b001', pb_dir=db_name)

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


def lpf(x, fc, fs):
    '''
    Function to implement low-pass filter
        Args: Input signal (x), cutoff frequency (fc), sampling frequency (fs)
        Returns: Filtered signal (x_filt)
    '''
    b, a = signal.butter(3, 2*fc/fs, btype='lowpass')
    x_filt = signal.lfilter(b,a,x)
    return x_filt


def hpf(x, fc, fs):
    '''
    Function to implement high-pass filter
        Args: Input signal (x), cutoff frequency (fc), sampling frequency (fs)
        Returns: Filtered signal (x_filt)
    '''
    b, a = signal.butter(3, 2*fc/fs, btype='highpass')
    x_filt = signal.lfilter(b,a,x)
    return x_filt


def bpf(x, lo, hi, fs):
    '''
    Function to implement band-pass filter
        Args: Input signal (x), cutoff frequencies (lo, hi), sampling frequency (fs)
        Returns: Filtered signal (x_filt)
    '''
    b = signal.firwin(30, [2*lo/fs, 2*hi/fs], pass_zero=False)
    x_filt = signal.lfilter(b,1,x)
    pdb.set_trace()
    # determine lowest butterworth order
    fc_lo = 2*lo/fs # lower cutoff frequency in rad
    fc_hi = 2*hi/fs # higher cutoff frequency in rad
    wp = [fc_lo, fc_hi] # passband edge frequencies
    ws = [fc_lo - fc_lo/2, fc_hi + fc_lo/2]
    gpass = 3 # passband gain
    gstop = 40 # stop band attenuation (negative)
    N, Wn = signal.buttord(wp, ws, gpass, gstop, analog=False)
    # generate coefficients and apply filter
    b, a = signal.butter(N, [fc_lo, fc_hi], btype='bandpass')
    x_filt = signal.filtfilt(b,a,x)
    return x_filt


def freq_analysis(y, fs):
    '''
    Analyze frequency content of input signal y to determine filter cutoffs
    '''
    out = {}

    # Fourier transform
    N = 8192 # number of fft bins
    xf = np.linspace( 0, fs/2, num=N//2 ) # frequency vector
    yf = fftpack.fft(y, n=N) # Fourier transform
    ps = (2/N) * np.abs(yf[0:int(N//2)]) # normalized power spectrum

    out['freq1'] = xf[np.argmax(yf)] # dominant frequency

    # energy in 5-15 Hz band

    plt.plot(xf, ps)
    plt.show()
    plt.close()

    return out


def remove_baseline_wander(x,y):
    '''
    Remove baseline wander by polynomial subtraction and/or high-pass filter
        Args: Time vector in sec (x) and ecg signal vector (y)
        Returns: Processed ecg signal vector (y_out)
    '''
    # plt.plot(x,y)

    # method 1: cubic spline fitting and subtraction
    t_range = [0,10] # process in 10-sec windows
    while t_range[1] < x[-1]:
        # extract segment of original signal
        idx = (x >= t_range[0]) & (x < t_range[1])
        dx = x[idx]
        dy = y[idx]
        # fit cubic function to segment
        p = np.polyfit(dx,dy,3)
        dy_fit = np.polyval(p,dx)
        # replace segment of original signal with spline-subtracted signal
        y[idx] = dy - dy_fit
        t_range = [k+8 for k in t_range] # increment time window w/ 20% overlap

    # method 2: high-pass filter

    '''
    plt.plot(x,y)
    plt.show()
    plt.close()
    '''

    y_out = y

    return y_out


def find_peaks():
    '''
    '''
    return 0



if __name__ == "__main__":

    # Combined measurement of ECG, breathing and seismocardiogram (CEBS database)

    # grab data from https://physionet.org/physiobank/database/cebsdb
    print('Loading signals from CEBS database...')
    data = get_data('cebsdb')

    

    pdb.set_trace()

    # TODO: loop all records in data list

    # plot measured ECG
    '''
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(data['t'], data['I'])
    # axarr[0].plot(data['t'], data['II'])
    '''

    print('Removing baseline wander...')
    ecg = remove_baseline_wander(data['t'],data['I'])

    # examine frequency content
    # freq_out = freq_analysis(data['I'], data['fs'])

    # for ECG, want to keep 5-15 Hz frequencies
    print('Band-pass filtering...')
    ecg_bp = bpf(ecg, 5, 15, data['fs'])

    plt.plot(ecg)
    plt.plot(ecg_bp)
    plt.show()

    pdb.set_trace()

    # derivative filter

    # absolute value

    axarr[0].legend(['Measured ECG', 'Processed ECG'])

    # line, = ax.plot([1, 2, 3], label='Inline label')


    # plot measured respiration
    axarr[1].plot(data['t'], data['RESP'])

    # TODO: plot derived respiration

    axarr[1].legend(['Measured ECG', 'Measured Respiration'])

    plt.show()
