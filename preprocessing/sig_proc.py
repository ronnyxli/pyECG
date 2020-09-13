
import numpy as np
from scipy import signal, fftpack

import pdb
from matplotlib import pyplot as plt

'''
Functions for signal processing
'''

def analysis(y,FS):
    '''
    Time-series and frequency analysis of input signal y
    '''

    # calculate PDF of signal
    # xf,Pyy = signal.welch(y, FS=FS, nperseg=2*FS, noverlap=FS)

    N = len(y)
    T = 1/FS # period
    yf = fftpack.fft(y - np.mean(y))
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    Pyy = np.abs(yf[0:N//2])**2

    out = {'freq':xf, 'pow':Pyy}

    # power calculations
    pow_tot = np.sum(Pyy)
    pow_5_50 = np.sum(Pyy[(xf >= 5) & (xf <= 50)])

    # SNR calculation
    out['5_50_ratio'] = pow_5_50/(pow_tot - pow_5_50)
    out['SNR'] = 20*np.log10(out['5_50_ratio'])

    return out


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
