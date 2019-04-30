
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
