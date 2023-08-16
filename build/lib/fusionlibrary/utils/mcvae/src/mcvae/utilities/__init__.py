import copy
import numpy as np
import random
import torch
import warnings
from sklearn.preprocessing import StandardScaler
from ..models.utils import DEVICE


def ltonumpy(X):
	"""

	:param X: list of pytorch variables or tensors
	:return:
	"""
	assert isinstance(X, list)
	assert len(X) > 0
	if isinstance(X[0], torch.Tensor):
		return [x.detach().numpy() for x in X]
	else:
		return [x.numpy() for x in X]


def ltotensor(X, device=DEVICE):
	"""

	:param X: list of numpy array or pytorch variables
	:return:
	"""
	assert isinstance(X, list)
	assert len(X) > 0
	ret = []
	for x in X:
		if isinstance(x, torch.Tensor):
			warnings.warn("List element is already a torch.Tensor")
			ret.append(x.clone())
		elif isinstance(x, np.ndarray):
			ret.append(torch.FloatTensor(x.copy()).to(device))
		elif x is None:
			ret.append(None)
		else:
			raise ValueError('Cannot transform element list to tensor')
	return ret
	# if isinstance(X[0], np.ndarray):
	# 	return [torch.FloatTensor(x).to(device) for x in X]


def preprocess_and_add_noise(X, snr, seed=0, device=DEVICE):
	"""

	:param X: list of pytorch variables
	:param snr:
	:param seed:
	:return:
	"""
	if not isinstance(snr, list):
		SNR = [snr for _ in X]
	else:
		SNR = snr

	X_ = ltonumpy(X)
	FIT = [StandardScaler().fit(x) for x in X_]
	X_std_ = [FIT[i].transform(X_[i]) for i in range(len(X_))]
	# X_std_ = [(x - x.mean(0)) / x.std(0, ddof=1) for x in X_]
	X_std = ltotensor(X_std_)

	# seed for reproducibility in training/testing based on prime number basis
	seed = seed + 3 * int(SNR[0] + 1) + 5 * len(X_) + 7 * X_[0].shape[0] + 11 * X_[0].shape[1]
	np.random.seed(seed)

	X_std_noisy_ = []
	for c, x in enumerate(X_std_):
		sigma_noise = np.sqrt(1.0/SNR[c])
		X_std_noisy_.append(x + sigma_noise * np.random.randn(*x.shape))

	X_std_noisy = ltotensor(X_std_noisy_, device=device)
	return X_std, X_std_noisy


def simulate_mar_multi_channel_data(x, intersection_fraction=0.5):
	"""
	Missing At Random data simulation
	N = f N + c n
	N: size of union
	f N: size of intersection
	c: n_channels
	n = unique id per channel
	"""
	f = intersection_fraction
	assert 0 <= f <= 1
	assert isinstance(x, list)
	N = x[0].shape[0]
	for xi in x:
		assert xi.shape[0] == N
	union = tuple(range(N))
	rng = random.Random(42)
	intersection = tuple(sorted(rng.sample(population=union, k=int(N * f))))

	assignments = []
	for c in range(len(x)):
		assignments.append(list(intersection))

	for i in union:
		if i not in intersection:
			c = rng.choice(range(len(x)))
			assignments[c].append(i)
	ids = tuple(tuple(sorted(a)) for a in assignments)
	x_m = [x[i][ids[i], ] for i in range(len(x))]
	return x_m, ids


def simulate_mnar_multi_channel_data(x, n_datasets, lambda_p=1, lambda_g=0):
	"""
	Missing Not At Random data simulation
	:param x: multi-channel dataset
	:param n_datasets: number of datasets where x is distributed
	:param lambda_p: pair-wise intersection. Every disjoint pair of datasets have "lambda_p" channels in common
	:param lambda_g: number of channels in common among every dataset.
	:return: x_mnar, ids
	"""
	assert isinstance(x, list)
	n_channels = len(x)
	assert n_datasets == n_channels
	assert lambda_g == n_channels or lambda_g == 0
	if lambda_g == n_channels:
		assert lambda_p == n_channels
	elif lambda_g == 0:
		assert lambda_p < n_channels - 1
	else:
		raise NotImplementedError

	N = x[0].shape[0]
	for xi in x:
		assert xi.shape[0] == N
	union = tuple(range(N))

	assignments = []
	for _ in range(n_channels):
		assignments.append([])

	rng = random.Random(42)
	for i in union:
		# select the dataset "i" belongs
		d = rng.choice(range(n_datasets))
		for c in range(n_channels):
			if lambda_g == n_channels:  # every dataset has all the channels
				assignments[c].append(i)
			elif lambda_g == 0:
				if c in [_ % n_channels for _ in range(d, d + lambda_p + 1)]:
					assignments[c].append(i)

	ids = tuple(tuple(sorted(a)) for a in assignments)
	x_mnar = [x[i][ids[i], ] for i in range(n_channels)]
	return x_mnar, ids


def lnormalize(X, FIT=None):
	assert isinstance(X, list)
	if FIT is None:
		FIT = [StandardScaler().fit(x) for x in X]
	X_std = [FIT[i].transform(X[i]) for i in range(len(X))]
	return X_std


__all__ = [
	'ltonumpy',
	'ltotensor',
	'preprocess_and_add_noise',
]
