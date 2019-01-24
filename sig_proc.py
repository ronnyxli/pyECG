
import numpy as np
from scipy import signal, fftpack

import pdb
from matplotlib import pyplot as plt

'''
Functions for signal processing
'''

def lpf(y, fc, N, fs):
    '''
    Function to implement low-pass filter
        Args: Input signal (y), cutoff frequency in Hz (fc), filter taps (N),
                sampling frequency in Hz (fs)
        Returns: Tuple containing filtered signal (y_filt) and filter coefficients (b)
    '''
    b = signal.firwin(numtaps=N, cutoff=2*fc/fs, window='hamming')
    y_filt = signal.lfilter(b,1,y)
    gd = int((N-1)/2) # calculate group delay
    return (y_filt[gd:], b)


def hpf(y, fc, N, fs):
    '''
    Function to implement high-pass filter
        Args: Input signal (y), cutoff frequency in Hz (fc), filter taps (N),
                sampling frequency in Hz (fs)
        Returns: Tuple containing filtered signal (y_filt) and filter coefficients (b)
    '''
    b = signal.firwin(numtaps=N, cutoff=2*fc/fs, pass_zero=False)
    y_filt = signal.lfilter(b,1,y)
    gd = int((N-1)/2) # calculate group delay
    return (y_filt[gd:], b)


def bpf(y, lo, hi, N, fs):
    '''
    Function to implement band-pass filter
        Args: Input signal (y), cutoff frequencies in Hz (lo, hi),
                filter taps (N), sampling frequency in Hz (fs)
        Returns: Filtered signal (y_filt) and filter coefficients (b)
    '''

    '''
    # determine lowest butterworth order to achieve desired
    fc_lo = 2*lo/fs # lower cutoff frequency in rad
    fc_hi = 2*hi/fs # higher cutoff frequency in rad
    wp = [fc_lo, fc_hi] # passband edge frequencies
    ws = [fc_lo - fc_lo/2, fc_hi + fc_lo/2]
    gpass = 3 # passband gain
    gstop = 40 # stop band attenuation (negative)
    N, Wn = signal.buttord(wp, ws, gpass, gstop, analog=False)

    # generate coefficients and apply filter
    b, a = signal.butter(N, [fc_lo, fc_hi], btype='bandpass')
    x_filt = signal.filtfilt(b,a,y)
    '''

    b = signal.firwin(30, [2*lo/fs, 2*hi/fs], pass_zero=False)
    y_filt = signal.lfilter(b,1,y)
    gd = int((N-1)/2) # calculate group delay
    return (y_filt[gd:], b)


def detrend(y, window_len):
    '''
    Remove baseline wander by polynomial subtraction
        Args: Input signal (y), window length in samples (window_len)
        Returns: None
    '''
    # process in windows of width specified by window_samp
    window_idx = np.arange(window_len)
    while window_idx[0] <= len(y):

        if window_idx[-1] > len(y):
            window_idx = np.arange(window_idx[0], len(y))

        sig = y[window_idx]

        # fit 4th-order polynomial
        p = np.polyfit(window_idx, sig, 4)
        sig_fit = np.polyval(p, window_idx)

        # replace original signal segment with de-trended signal segment
        np.put(y, window_idx, sig-sig_fit)

        # shift window
        window_idx += window_len

    return 0




def proc_ecg(ecg, fs):
    '''
    Execute processing steps for input ECG signal (ecg) with sampling frequency fs
    '''
    # low-pass filter
    ecg, lpf_coeff = lpf(ecg, 30, 31, fs)

    # x, b = lpf(sig, 0.5, 31, fs)
    # sig_fit = signal.filtfilt(b,1,sig)

    # baseline removal
    detrend(ecg, 2*fs)

    return ecg
