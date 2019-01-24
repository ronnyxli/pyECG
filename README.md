# pyECG
This project contains functions for processing, compressing, and extracting physiological features from ECG signals.

The script main.py tests the functions on sample data from the Fantasia database on Physionet, downloaded using the wfdb python package:
record = wfdb.rdsamp('f1o01', pb_dir='fantasia')

Required Python modules:
- numpy (for quantitative operations)
- scipy (for signal processing)
- matplotlib (for visualization)
- pywt (for wavelet operations)

###### Metric 1: Heart rate
(In progress)

###### Metric 2: Heart rate variability
(In progress)

###### Metric 3: ECG-derived respiration
(In progress)

######

###### References:
1) Pan J and Tompkins WJ. "A Real-Time QRS Detection Algorithm." IEEE Trans Biomed Eng. 1985;32(3): 232-6.
