
# data I/O
import os
import wfdb
import scipy.io as sio
import pickle

# quantitative
import numpy as np

# signal processing
from scipy import signal, fftpack

# plotting
from matplotlib import pyplot as plt
from matplotlib import gridspec

from compression import compress
from reconstruction import reconstruct
# from calc_ecg_features import detect_R_peaks

# debugging
import pdb

# physionet database
db_name = 'mitdb' # 'cdb'

PLOT = True



def analysis(y,fs):
    '''
    Time-series and frequency analysis of input signal
        Args: y = input signal, fs = sampling frequency
        Out: Time and frequency domain features of signal y
    '''

    out = {}

    # calculate PDF using 4-second windows with 50% overlap
    f,Pyy = signal.welch(y, fs=fs, nperseg=2*fs, noverlap=fs)

    plt.subplot(211)
    plt.plot(y)
    plt.grid(True)

    plt.subplot(212)
    plt.plot(f, Pyy)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True)

    plt.show()
    plt.close()

    out['max_freq'] = f[np.argmax(Pyy)] # dominant frequency

    # TODO: calculate energy ratio in 5-15 Hz and 5-40 Hz bands

    # histogram distribution
    y_counts, y_bins = np.histogram(y)

    return out


def calc_PRD(xs,xr):
    '''
    original signal (xs) and resconstructed signal (xr)
    '''
    MSE = np.sum( (xs - xr)**2 )
    return 100*np.sqrt( MSE/np.sum([float(x)**2 for x in xs]) )



if __name__ == "__main__":

    # loop all records in database
    for record in wfdb.get_record_list(db_name, records='all'):

        # get data for current record
        data = wfdb.rdsamp(record, pb_dir=db_name)
        fs = data[1]['fs']
        ecg = data[0][:,0][0:fs*20]

        header, wc_compressed, wc_orig = compress(ecg)

        ecg_recon, wc_recon = reconstruct(header, wc_compressed)

        PRD = calc_PRD(ecg, ecg_recon)

        if PLOT:

            fig = plt.figure(figsize=(15,8))

            # plot reconstructed wavelet coefficients with original
            gs = gridspec.GridSpec(12,11)

            # ECG signal
            ax0 = plt.subplot(gs[0:4, 0:11])
            ax0.plot(ecg)
            ax0.plot(ecg_recon)
            ax0.grid(True)
            # plt.title('ECG Signal')
            plt.title( 'CR = ' + str(round(header['CR'],3)) + ', PRD = '  + str(round(PRD,3)) )

            # level 1 detail (cD1)
            ax1 = plt.subplot(gs[5:8, 0:3])
            ax1.plot(wc_orig[5])
            ax1.plot(wc_recon[5])
            ax1.grid(True)
            plt.title('cD1 (64-128 Hz)')

            # level 2 detail (cD2)
            ax2 = plt.subplot(gs[5:8, 4:7])
            ax2.plot(wc_orig[4])
            ax2.plot(wc_recon[4])
            ax2.grid(True)
            plt.title('cD2 (32-64 Hz)')

            # level 3 detail (cD3)
            ax3 = plt.subplot(gs[5:8, 8:11])
            ax3.plot(wc_orig[3])
            ax3.plot(wc_recon[3])
            ax3.grid(True)
            plt.title('cD3 (16-32 Hz)')

            # level 4 detail (cD4)
            ax4 = plt.subplot(gs[9:12, 0:3])
            ax4.plot(wc_orig[2])
            ax4.plot(wc_recon[2])
            ax4.grid(True)
            plt.title('cD4 (8-16 Hz)')

            # level 5 detail (cD5)
            ax5 = plt.subplot(gs[9:12, 4:7])
            ax5.plot(wc_orig[1])
            ax5.plot(wc_recon[1])
            ax5.grid(True)
            plt.title('cD5 (4-8 Hz)')

            # level 5 approximation (cA5)
            ax5 = plt.subplot(gs[9:12, 8:11])
            ax5.plot(wc_orig[0])
            ax5.plot(wc_recon[0])
            ax5.grid(True)
            plt.title('cA5 (0-4 Hz)')

            plt.show()
            plt.close()

            pdb.set_trace()

        '''
        try:
            # get annotations for current record
            ann = wfdb.rdann(record, extension='atr', pb_dir=db_name)
            ann.symbol
            ann.sample
        except:
            print('Failed to get annotations')
        '''
