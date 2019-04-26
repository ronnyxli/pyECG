
'''
Functions to implement wavelet-based signal reconstruction
'''

import numpy as np
import pywt

import pdb

# compression parameters
WAVELET_TYPE    = 'db4'
WAVEDEC_LEVELS  = 5
QUANT_PRECISION = 8 # bits
FILT_TAPS       = 8

# define FIR low-pass and high-pass filters for wavelet reconstruction
# LO_R = np.array([0.2304, 0.7148, 0.6309,-0.0280,-0.1870, 0.0308, 0.0329, -0.0106])*np.sqrt(2)
# HI_R = np.array([-0.0106, -0.0329, 0.0308, 0.1870, -0.0280, -0.6309, 0.7148, -0.2304])*np.sqrt(2)
LO_R = np.array([ 0.32583, 1.01088, 0.89222, -0.039600, -0.26446, 0.04356, 0.04653, -0.01500])
HI_R = np.array([-0.01500, -0.04653,  0.04356,  0.26446, -0.03960, -0.89222, 1.01088, -0.32583])


def rescale(x, B, lims):
    '''
    Re-scale values in x to
        Args: input list (x) with B-bit precision and original limits (lims)
        Returns: new np array with re-scaled values (xs)
    '''

    quant_range = np.linspace(0, 2**B-1, 2**B)
    bin_range = np.linspace(lims[0], lims[1], 2**B)
    xs = []
    for s in x:
        # map to original bin limits (reverse quantize)
        idx = np.where(s == quant_range)
        xs.append(bin_range[idx[0][0]])

    return np.array(xs)


def reconstruct(hdr, coeffs):
    '''
        Args:
        Returns: reconstructed signal (y_recon), detail coefficients, length of
            original signal (L0)
    '''

    # store reconstructed wavelet coefficients as list of numpy arrays
    coeffs_recon = []

    # retained all cA5 coefficients
    cA = coeffs[0:hdr['c_lengths'][0]]
    coeffs_recon.append(rescale(cA, QUANT_PRECISION, hdr['cA_lims']))

    # retained select cD1-5 coefficients
    cD_rescaled = rescale(coeffs[hdr['c_lengths'][0]:], QUANT_PRECISION, hdr['cD_lims'])
    cD_idx = -1
    for n in range(1,len(hdr['c_lengths'])):
        d_length = hdr['c_lengths'][n]
        cD = [0]*d_length
        if n == 1:
            start_idx = 0
        else:
            start_idx = np.sum(hdr['c_lengths'][1:n])
        sig_idx = hdr['ind'][start_idx:start_idx+d_length]
        for m in range(0,len(sig_idx)):
            if sig_idx[m]:
                cD_idx += 1
                cD[m] = cD_rescaled[cD_idx]
        coeffs_recon.append(np.array(cD))

    y_recon = pywt.waverec(coeffs_recon, WAVELET_TYPE)

    return y_recon, coeffs_recon
