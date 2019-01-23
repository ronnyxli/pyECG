
import numpy as np
from scipy import signal, fftpack

import pdb

'''
Functions for signal processing
'''



def lpf(y, fc, N, fs):
    '''
    Function to implement low-pass filter
        Args: Input signal (y), cutoff frequency (fc), filter length (N), sampling frequency (fs)
        Returns: Filtered signal (y_filt)
    '''
    b = signal.firwin(numtaps=N, cutoff=fc, fs=fs)
    y_filt = signal.lfilter(b,1,y)
    gd = int((N-1)/2) # calculate group delay
    return y_filt[gd-1:]


def hpf(y, fc, N, fs):
    '''
    Function to implement high-pass filter
        Args: Input signal (y), cutoff frequency (fc), filter order (N), sampling frequency (fs)
        Returns: Filtered signal (y_filt)
    '''
    b = signal.firwin(numtaps=N, cutoff=fc, fs=fs, pass_zero=False)
    y_filt = signal.lfilter(b,1,y)
    gd = int((N-1)/2) # calculate group delay
    return y_filt[gd-1:]


def bpf(y, lo, hi, fs):
    '''
    Function to implement band-pass filter
        Args: Input signal (x), cutoff frequencies (lo, hi), sampling frequency (fs)
        Returns: Filtered signal (x_filt)
    '''
    b = signal.firwin(30, [2*lo/fs, 2*hi/fs], pass_zero=False)
    y_filt = signal.lfilter(b,1,y)
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
    x_filt = signal.filtfilt(b,a,y)

    return x_filt


def remove_baseline(y,win_length):
    '''
    Remove baseline wander by polynomial subtraction and/or high-pass filter
        Args: y = input signal, win_length = number of samples in processing segments
        Returns: Processed signal vector (y_out)
    '''
    # cubic spline fitting and subtraction
    t_range = [0,10] # process in 10-sec windows
    while t_range[1] < y[-1]:
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

    y_out = y

    return y_out


def find_peaks():
    '''
    '''
    return 0
