
'''
Functions to implement wavelet-based compression of a 1D signal
'''

import numpy as np
import pywt
import itertools
from scipy import signal

from matplotlib import pyplot as plt

import pdb

# compression parameters
WAVELET_TYPE    = 'db4'
WAVEDEC_LEVELS  = 5
QUANT_PRECISION = 8 # bits
FILT_TAPS       = 8

# signal parameters
SIG_RES = 12 # bits

# define FIR low-pass and high-pass filters for wavelet decomposition
# LO_D = np.array([-0.0106, 0.0329, 0.0308, -0.1870, -0.0280, 0.6309, 0.7148, 0.2304])/np.sqrt(2)
# HI_D = np.array([-0.2304, 0.7148, -0.6309, -0.0280, 0.1870, 0.0308, -0.0329, -0.0106])/np.sqrt(2)
LO_D = np.array([-0.00750, 0.02326, 0.02178, -0.13223, -0.01980, 0.44611, 0.50544, 0.16292])
HI_D = np.array([-0.16292, 0.50544, -0.44611, -0.01980, 0.13223, 0.02178, -0.02326, -0.00750])



def stabilize_baseline(x, fs):
    '''
    '''
    fc = 1 # cutoff frequency (Hz) for LPF
    b = signal.firwin(numtaps=21, cutoff=2*fc/fs, window='hamming')
    # b = signal.firwin(numtaps=21, cutoff=2*fc/fs, pass_zero=False) # HPF
    x_filt = signal.filtfilt(b,1,x)
    '''
    plt.plot(x, label='cA5')
    plt.plot(x_filt, label='1 Hz low-passed')
    plt.plot(x - x_filt, label='Adjusted cA5')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()
    pdb.set_trace()
    '''
    return x - x_filt


def quantize(x, B):
    '''
    Quantize values in array x to B-bit precision
        Args: input array (x) and bit-precision specified by B
        Returns: new array with quantized values (xq) and limits of original
            data (x_lims)
    '''
    quant_range = np.linspace(0, 2**B-1, 2**B)

    # compute histogram of original data
    bin_counts, bin_edges = np.histogram(x, bins=2**B)

    xq = np.zeros(len(x))
    for n in range(1,len(bin_edges)):
        lo = bin_edges[n-1]
        hi = bin_edges[n]
        xq[(x >= lo) & (x <= hi)] = quant_range[n-1]

    x_lims = [bin_edges[0], bin_edges[-1]]
    '''
    plt.subplot(211)
    plt.plot(x)
    plt.grid(True)
    plt.subplot(212)
    plt.hist(x,bins=25)
    plt.grid(True)
    plt.show()
    plt.close()

    plt.subplot(211)
    plt.plot(xq)
    plt.grid(True)
    plt.subplot(212)
    plt.hist(xq,bins=25)
    plt.grid(True)
    plt.show()
    plt.close()

    pdb.set_trace()
    '''
    return xq, x_lims


def encode(x):
    '''
    '''
    pdb.set_trace()
    return True


def adaptive_thresh(x):
    '''
    Selectively discard (i.e. zero-out) the non-significant coefficients
        Args: Input array (x)
        Returns: Filtered array containing only significant elements (out),
            binary array equal to length of x indicating significant
            indexes (idx), and maximum distance used to establish thresh value
    '''

    out = np.zeros(len(x))

    # based on detecting elbow in x_desc

    # sort absolute values in descending order and normalize by max
    x_desc = sorted(np.abs(x), reverse=True)
    # x_desc = x_desc/np.max(x_desc)

    # define line from first point to last point (Ax + By + C = 0)
    A = (x_desc[0] - x_desc[-1])/len(x)
    B = 1
    C = -x_desc[0]

    # loop each point along descending curve and calculate shortest distance to line
    dist_vec = []
    for n in range(0,len(x)):
        px = n
        py = x_desc[n]
        d = np.abs(A*px + B*py + C)/np.sqrt(A**2 + B**2)
        dist_vec.append(d)

    # find index of maximum distance
    xm = np.argmax(dist_vec)
    ym = x_desc[xm] # this is the threshold value

    # find coordinates of point along line
    dx = (-C-B*ym-A*xm)/(A*(1+B))
    dy = (-C-A*dx)/B

    # get corresponding threshold
    thresh = ym

    # get indexes of significant coefficients to retain
    idx = (np.abs(x) > ym).tolist()

    # zero-out nonsignificant coefficients
    out = list(itertools.compress(x, idx))
    '''
    plt.subplot(221)
    plt.plot(x_desc)
    plt.plot([0,len(x)], [x_desc[0],x_desc[-1]], 'r--')
    plt.plot( [xm,dx], [ym,dy], 'g--' )
    plt.grid(True)
    plt.xlabel('Index')
    plt.ylabel('|Coefficient|')

    plt.subplot(222)
    # plt.hist(x, bins='auto')
    plt.plot(dist_vec)
    plt.xlabel('Index')
    plt.ylabel('Shortest distance from line')
    plt.grid(True)

    plt.subplot(223)
    plt.plot(x)
    plt.grid(True)

    plt.subplot(224)
    plt.plot(x)
    plt.plot(x*idx)
    plt.plot([0,len(x)], [thresh,thresh], 'r--')
    plt.plot([0,len(x)], [-thresh,-thresh], 'r--')
    plt.grid(True)

    plt.show()
    plt.close()
    pdb.set_trace()
    '''
    return out, idx


def compress(sig, fs):
    '''
    Perform compression on input signal x using compression parameters
        Args:
        - Input signal (sig)
        - Sampling frequency (fs)
        Returns:
        - Compression ratio
        - list of dicts containing compressed wavelet coefficients
        - original wavelet coefficients
    '''

    # list of dicts to store compressed wavelet coefficients
    compressed_coeffs = []

    # wavelet decomposition using PyWavelets
    wavelet_coeffs = pywt.wavedec(sig, WAVELET_TYPE, level=WAVEDEC_LEVELS)

    cA = stabilize_baseline(wavelet_coeffs[0], fs/2**(WAVEDEC_LEVELS))
    # cA = wavelet_coeffs[0]

    # quantize lowest level approximation coeffs
    cA_quant, cA_lims = quantize(cA, QUANT_PRECISION)

    # create dict for approximation coeffs and append to list of compressed coeffs
    size = len(cA_quant)*QUANT_PRECISION + len(cA_lims)*SIG_RES # number of bits required to store
    compressed_coeffs.append( {'size':size, 'val':cA_quant,\
                                'lims':cA_lims,\
                                'ind':[1]*len(wavelet_coeffs[0])} )

    # selectively discard detail coefficients at each level and remember indexes to retain
    for k in range(1,len(wavelet_coeffs)):
        if k == 1:
            # retain all lowest level detail coefficients
            ind = [1]*len(wavelet_coeffs[k])
            cD = list(wavelet_coeffs[k])
        elif k == len(wavelet_coeffs) - 1:
            # discard all highest level detail coefficients
            ind = [0]*len(wavelet_coeffs[k])
            cD = []
        else:
            # selectively retain detail coefficients
            cD, ind = adaptive_thresh(wavelet_coeffs[k])
            # TODO: determine when to discard all
            # ind = [0]*len(wavelet_coeffs[k])
            # cD = []

        # quantize retained detail coefficients
        if len(cD) > 0:
            cD_quant, cD_lims = quantize(cD, QUANT_PRECISION)
            size = len(cD_quant)*QUANT_PRECISION + len(cD_lims)*SIG_RES
            if k > 1:
                size += len(ind) # add number of bits equal length of coeffs
        else:
            cD_quant = []
            cD_lims = []
            size = 0

        # append to list of compressed coeffs
        compressed_coeffs.append( {'size':size, 'val':cD_quant, 'lims':cD_lims, 'ind':ind} )

    # TODO: encode
    # cD_enc = encode(cD_quant)

    # calculate compression ratio
    original_size = len(sig)*SIG_RES # bits
    compressed_size = sum([x['size'] for x in compressed_coeffs])
    compression_ratio = original_size/compressed_size

    return compression_ratio, compressed_coeffs, wavelet_coeffs
