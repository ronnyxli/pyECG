
# data I/O
import os
import wfdb
import scipy.io as sio
import pickle

# quantitative
import numpy as np

# signal processing
from scipy import signal, stats, fftpack

# plotting
from matplotlib import pyplot as plt
from matplotlib import gridspec

from compression import compress
from reconstruction import reconstruct

# debugging
import pdb

PLOT = False


def calc_PRD(xs,xr):
    # percent RMS difference
    MSE = np.sum( (xs - xr)**2 )
    # return PRD and Pearson's correlation (R)
    pearson_corr = stats.pearsonr(xs,xr)
    return 100*np.sqrt( MSE/np.sum([float(x)**2 for x in xs]) ), pearson_corr[0]


def analysis(y,fs):
    '''
    Time-series and frequency analysis of input signal y
    '''

    # calculate PDF of signal
    # xf,Pyy = signal.welch(y, fs=fs, nperseg=2*fs, noverlap=fs)

    N = len(y)
    T = 1/fs # period
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



if __name__ == "__main__":

    print('Change made in develop branch')

    # initialize empty lists for evaluation metrics
    CR_arr = []
    PRD_arr = []
    R_arr = []
    SNRo_arr = []
    SNRr_arr = []
    d_SNR = []

    # loop all records in database
    for record in wfdb.get_record_list(db_name, records='all'):

        # record = 'Person_01/rec_10'
        # record = '203'

        # get data for current record
        data = wfdb.rdsamp(record, pb_dir=db_name + '/' + record.split('/')[0])

        Fs = data[1]['fs']
        ecg = data[0][:,0]#[0:Fs*20]

        # zero-mean
        ecg = ecg - np.mean(ecg)

        # call compression function
        CR, ecg_compressed, wc_orig = compress(ecg, Fs)

        # call reconstruction function
        ecg_recon, wc_recon = reconstruct(ecg_compressed)

        # compute and store evaluation metrics
        PRD, R = calc_PRD(ecg, ecg_recon)
        ps = analysis(ecg, Fs)
        pr = analysis(ecg_recon, Fs)
        CR_arr.append(CR)
        PRD_arr.append(PRD)
        R_arr.append(R)
        SNRo_arr.append(ps['SNR'])
        SNRr_arr.append(pr['SNR'])
        d_SNR.append(pr['SNR'] - ps['SNR'])

        print( record + ': ' + 'CR = ' + str(round(CR,3)) + '; PRD = ' + str(round(PRD,3)) + '; R = ' + str(round(R,3)) )
        print( record + ': ' + 'SNR orig = ' + str(round(ps['SNR'],3)) + '; SNR recon = ' + str(round(pr['SNR'],3)) )

        '''
        fig = plt.figure(figsize=(12,6))
        plt.subplot(211)
        plt.plot(ecg)
        plt.title('SNR = ' + str(ps['SNR']))
        # plot PDF
        plt.subplot(212)
        plt.plot(ps['freq'],ps['pow'])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.grid(True)

        plt.show()
        plt.close()

        pdb.set_trace()
        '''
        if PLOT & (CR > 9) & (R < 0.5):

            fig = plt.figure(figsize=(15,8))

            # plot reconstructed wavelet coefficients with original
            gs = gridspec.GridSpec(12,11)

            # ECG signal
            ax0 = plt.subplot(gs[0:4, 0:11])
            ax0.plot(ecg)
            ax0.plot(ecg_recon)
            ax0.grid(True)
            plt.title('ECG Signal (0-' + str(Fs/2) + ' Hz)')
            # plt.title( 'CR = ' + str(round(CR,3)) + ', PRD = '  + str(round(PRD,3)) )

            # level 1 detail (cD1)
            ax1 = plt.subplot(gs[5:8, 0:3])
            ax1.plot(wc_orig[5])
            ax1.plot(wc_recon[5])
            ax1.grid(True)
            plt.title('cD1 (' + str(Fs/4) + '-' + str(Fs/2) + ' Hz)')

            # level 2 detail (cD2)
            ax2 = plt.subplot(gs[5:8, 4:7])
            ax2.plot(wc_orig[4])
            ax2.plot(wc_recon[4])
            ax2.grid(True)
            plt.title('cD2 (' + str(Fs/8) + '-' + str(Fs/4) + ' Hz)')

            # level 3 detail (cD3)
            ax3 = plt.subplot(gs[5:8, 8:11])
            ax3.plot(wc_orig[3])
            ax3.plot(wc_recon[3])
            ax3.grid(True)
            plt.title('cD3 (' + str(Fs/16) + '-' + str(Fs/8) + ' Hz)')

            # level 4 detail (cD4)
            ax4 = plt.subplot(gs[9:12, 0:3])
            ax4.plot(wc_orig[2])
            ax4.plot(wc_recon[2])
            ax4.grid(True)
            plt.title('cD4 (' + str(Fs/32) + '-' + str(Fs/16) + ' Hz)')

            # level 5 detail (cD5)
            ax5 = plt.subplot(gs[9:12, 4:7])
            ax5.plot(wc_orig[1])
            ax5.plot(wc_recon[1])
            ax5.grid(True)
            plt.title('cD5 (' + str(Fs/64) + '-' + str(Fs/32) + ' Hz)')

            # level 5 approximation (cA5)
            ax5 = plt.subplot(gs[9:12, 8:11])
            ax5.plot(wc_orig[0])
            ax5.plot(wc_recon[0])
            ax5.grid(True)
            plt.title('cA5 (0-' + str(Fs/64) + ' Hz)')

            plt.show()
            plt.close()

            pdb.set_trace()

        '''
        try:
            # get annotations for current record
            ann = wfdb.rdann(record, extension='atr', pb_dir=db_name)

            # plot original ECG with annotated R-peaks
            plt.subplot(211)
            plt.plot(ecg)
            plt.grid(True)

            for n in range(0,len(ann.symbol)):
                if ann.sample[n] > len(ecg):
                    break
                plt.plot(ann.sample[n], ecg[ann.sample[n]], 'rx')

            # plot reconstructed ECG with detected R-peaks
            plt.subplot(212)
            plt.plot(ecg_recon)
            plt.grid(True)

            plt.show()
            plt.close()

            pdb.set_trace()


        except:
            print('Failed to get annotations')
        '''

    # box plot of SNR
    plt.boxplot([SNRo_arr, SNRr_arr], showmeans=True, labels=['Original', 'Reconstructed'])
    plt.grid(True)
    plt.ylabel('SNR (dB)')

    plt.show()
    plt.close()

    plt.plot(CR_arr, PRD_arr, 'b.')
    plt.grid(True)
    plt.xlabel('CR')
    plt.ylabel('PRD')

    plt.show()
    plt.close()

    plt.plot(CR_arr, R_arr, 'b.')
    plt.grid(True)
    plt.xlabel('CR')
    plt.ylabel('R')

    plt.show()
    plt.close()

    pdb.set_trace()
