
import sys

import numpy as np
from scipy import signal, fftpack
import pywt

from matplotlib import pyplot as plt
import pdb

'''
Functions to compress a 1D signal
'''

def thresh_sum(b, p):
    '''
    Establish threshold value on array b based on p% of absolute cumulative sum
    '''
    cs_thresh = p * np.sum(abs(b))
    cs = np.cumsum(abs(b))
    idx = np.argmax(cs > cs_thresh)
    b_desc = sorted(abs(b), reverse=True)
    thresh = b_desc[idx]
    return thresh


def quantize(b, p):
    '''
    Maps distribution of array b within bounds [0 : (2^p)-1]
    '''

    bin_counts, bin_edges = np.histogram(b, bins=2**p)
    quant_range = np.arange(2**p)

    # len(bin_edges) = len(quant_range) - 1

    # initialize array of 0's to store quantized coefficients
    bq = np.repeat(0, len(b))

    # loop all bins in histogram
    for n in range(1,len(bin_edges)):
        lo = bin_edges[n-1]
        hi = bin_edges[n]
        idx = np.argwhere((b >= lo) & (b <= hi))
        if len(idx) > 0:
            np.put(bq, idx, len(idx)*[quant_range[n-1]])

    '''
    plt.hist(b, bins=2**p)
    plt.hist(bq, bins=2**p)
    plt.show()
    plt.close()
    '''

    return bq, [bin_edges[0], bin_edges[-1]]



def compress(x,params):
    '''
    Args: x = input vector, params = dictionary of compression parameters
    '''

    x_norm = x - np.mean(x)

    # discrete cosine transform
    coeff = fftpack.dct(x_norm)
    # coeff = pywt.wavedec(x, 'db1', level=2)

    # threshold coefficients
    coeff_thresh = thresh_sum(coeff, params['energyThresh'])

    # find last index of coefficients to retain
    cutoff_idx = np.max(np.where(abs(coeff) > coeff_thresh))

    coeff_quant, bin_lims  = quantize(coeff[0:cutoff_idx], params['quantPrecision'])

    # calculate compression ratio as ratio of original to compressed size in bytes
    orig_size = len(x) * sys.getsizeof(x[0])
    compressed_size = len(coeff_quant) * params['quantPrecision']/8 + \
                        sys.getsizeof(cutoff_idx) + sys.getsizeof(bin_lims)
    CR = orig_size/compressed_size

    out = {'coeff_quant':coeff_quant, 'cutoff_idx':cutoff_idx, 'bin_lims':bin_lims, 'CR':CR}

    return out




def reconstruct(x):
    '''
    '''

    # inverse DCT
    x_recon = fftpack.idct(b)

    return True
