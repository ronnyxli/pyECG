

def find_peaks():
    '''
    '''
    return 0


def calc_rr(ecg,fs):
    '''
    Calculate R-R intervals from ECG signal
        Args: ECG segment (ecg), sampling frequency (fs)
        Returns: rr_vec = list of tuples where each tuple represents the index
            and amplitude of a R-peak
    '''

    rr_vec = []

    # differentiator filter

    # rectification

    plt.plot(sig)
    plt.plot(sig_hpf)
    plt.show()
    plt.close()

    pk_idx,pk_amp = find_peaks(ecg_hpf)

    # add to RR array
    for n in range(0,len(pk_idx)):
        rr_vec.append( (pk_idx[n],pk_amp[n]) )

    return rr_vec


def calc_hrv():
    '''
    '''
    return  True


def calc_resp():
    '''
    '''
    return  True
