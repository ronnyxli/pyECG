
import sys

import numpy as np
from scipy import signal, fftpack
import pywt

from matplotlib import pyplot as plt
import pdb

'''
Functions for signal compression
'''


def compress(x):
    '''
    '''
    orig_size = len(x) * sys.getsizeof(x[0]) # bytes

    b = pywt.wavedec(x, 'db1', level=2)
    pdb.set_trace()

    # discrete cosine transform
    b = fftpack.dct(x)



    return True

def reconstruct(x):
    '''
    '''

    # inverse DCT
    x_recon = fftpack.idct(b)

    return True
