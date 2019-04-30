
'''
Functions to implement wavelet-based compression of a 1D signal
'''

import numpy as np
import pywt
import itertools

from matplotlib import pyplot as plt

import pdb

# compression parameters
WAVELET_TYPE    = 'db4'
WAVEDEC_LEVELS  = 5
QUANT_PRECISION = 8 # bits
FILT_TAPS       = 8

# define FIR low-pass and high-pass filters for wavelet decomposition
# LO_D = np.array([-0.0106, 0.0329, 0.0308, -0.1870, -0.0280, 0.6309, 0.7148, 0.2304])/np.sqrt(2)
# HI_D = np.array([-0.2304, 0.7148, -0.6309, -0.0280, 0.1870, 0.0308, -0.0329, -0.0106])/np.sqrt(2)
LO_D = np.array([-0.00750, 0.02326, 0.02178, -0.13223, -0.01980, 0.44611, 0.50544, 0.16292])
HI_D = np.array([-0.16292, 0.50544, -0.44611, -0.01980, 0.13223, 0.02178, -0.02326, -0.00750])




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

    # sort absolute values in descending order
    x_desc = sorted(np.abs(x), reverse=True)

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
    '''
    return out, idx


def compress(sig):
    '''
    Perform compression on input signal x using compression parameters
        Args: input signal (sig)
        Returns: tuple containing header and encoded wavelet coefficients
            - header containing length of coefficients, bin limits,
                indexes of coeffs to retain, and compression ratio
            - encoded wavelet coefficients
            - original wavelet coefficients
            - quality metrics for input signal
    '''

    # wavelet decomposition using PyWavelets
    wavelet_coeffs = pywt.wavedec(sig, WAVELET_TYPE, level=WAVEDEC_LEVELS)

    # quantize lowest level approximation coeffs
    cA_quant, cA_lims = quantize(wavelet_coeffs[0], QUANT_PRECISION)

    # selectively discard detail coefficients at each level and keep indexes to retain
    cD = []
    sig_idx = []
    for k in range(1,len(wavelet_coeffs)):
        if k == 1:
            # retain all lowest level detail coefficients
            ind = [1]* len(wavelet_coeffs[k])
            cD_new = list(wavelet_coeffs[k])
        elif k == len(wavelet_coeffs) - 1:
            # discard all highest level detail coefficients
            ind = [0]* len(wavelet_coeffs[k])
            cD_new = []
        else:
            # selectively retain detail coefficients
            cD_new, ind = adaptive_thresh(wavelet_coeffs[k])
            # TODO: determine when to discard all
            # ind = [0]* len(wavelet_coeffs[k])
            # cD_new = []
            
        cD = cD + cD_new
        sig_idx = sig_idx + ind

    # quantize retained detail coefficients from all levels
    cD_quant, cD_lims = quantize(cD, QUANT_PRECISION)

    # TODO: encode
    # cD_enc = encode(cD_quant)

    # calculate compression ratio
    original_size = len(sig)*32 # bits
    compressed_size = (len(cA_quant) + len(cD_quant))*QUANT_PRECISION\
                        + (len(cA_lims) + len(cD_lims))*32 + len(sig_idx)\
                        - len(wavelet_coeffs[-1]) - len(wavelet_coeffs[1]) # bits
    CR = original_size/compressed_size

    hdr = {'c_lengths':[len(x) for x in wavelet_coeffs], 'cA_lims':cA_lims,\
                'cD_lims':cD_lims,'ind':[int(x) for x in sig_idx],'CR':CR}

    return hdr, list(itertools.chain(cA_quant, cD_quant)), wavelet_coeffs
