
import sys

import numpy as np
from scipy import signal, fftpack
import pywt

from matplotlib import pyplot as plt
import pdb

'''
Functions to compress a 1D signal
'''

def thresh_cumsum(b, p):
    '''
    Establish threshold value on array b based on p% of cumulative sum
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

    b_hist = np.histogram(b, bins=2**p)
    quant_range = np.arange(2**p)
    # for x in b:

    return True



def compress(x,params):
    '''
    Args: x = input vector, params = dictionary of compression parameters
    '''

    # calculate size of original signal in bytes
    orig_size = len(x) * sys.getsizeof(x[0]) # bytes

    x_norm = x - np.mean(x)

    # discrete cosine transform
    coeff = fftpack.dct(x_norm)
    # coeff = pywt.wavedec(x, 'db1', level=2)

    # threshold coefficients
    coeff_thresh = thresh_cumsum(coeff, params['energyThresh'])

    # find last index of coefficients to retain
    cutoff_idx = np.max(np.where(abs(coeff) > coeff_thresh))

    bin_lims, coeff_quant = quantize(coeff[0:cutoff_idx], params['quantPrecision'])

    # calculate size of compressed signal in bytes
    compressed_size = len(coeff_quant) * params['quantPrecision']/8 + \
                        sys.getsizeof(cutoff_idx) + sys.getsizeof(bin_lims)

    pdb.set_trace()


    return True




def reconstruct(x):
    '''
    '''

    # inverse DCT
    x_recon = fftpack.idct(b)

    return True
