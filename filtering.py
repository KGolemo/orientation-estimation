import scipy
import numpy as np


def medianFilter(data, f_size):
	n_samples, n_signals = data.shape
	f_data = np.zeros([n_samples, n_signals])
	for i in range(n_signals):
		f_data[:, i] = scipy.signal.medfilt(data[:, i], f_size)
	return f_data

def lowpassFilter(data, cutoff, fs, order):
	b, a = scipy.signal.butter(order, cutoff, fs=fs, btype='low', analog=False)
	n_samples, n_signals = data.shape
	f_data = np.zeros([n_samples, n_signals])
	for i in range(n_signals):
		f_data[:, i] = scipy.signal.lfilter(b, a, data[:, i])
	return f_data
