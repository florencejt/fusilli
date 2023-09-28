import numpy as np
import torch
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt


def multiply_gaussians(mean1, var1, mean2, var2):
	"""
	The Product of two Gaussian PDFs is a Scaled (un-normalized) Gaussian PDF.
	Here we provide only the proper (normalized) PDF.
	"""
	mean = (mean1 * var2 + mean2 * var1) / (var1 + var2)
	var = 1 / (1 / var1 + 1 / var2)

	return mean, var


def divide_gaussians(mean1, var1, mean2, var2):
	"""
	gauss1 / gauss2
	"""
	var = 1 / (1 / var1 - 1 / var2)
	mean = mean1 + var * (1 / var2) * (mean1 - mean2)

	return mean, var


def cholesky_to_variance(diagonal, lower_diagonal):
	"""
	Return Sigma = CC', where C is che Cholesky factor
	"""
	L = len(diagonal)
	C = torch.diag(diagonal)
	C[torch.ones(L, L).tril(-1) == 1] = lower_diagonal
	return torch.matmul(C, C.t())


def plot_covariance_ellipsoid(sigma):
	t = np.linspace(0, 2 * np.pi)
	if not isinstance(sigma, list):
		sigma = [sigma]
	for sigma_ in sigma:
		d, v = np.linalg.eig(sigma_)
		print(f'Sqrt(eig_values) = {d ** 0.5}')
		x = np.array([[np.cos(_), np.sin(_)] for _ in t])
		a = (v*d**0.5).dot(x.T).T
		plt.plot(a[:, 1], a[:, 0])
	plt.xlabel('ax 1')
	plt.ylabel('ax 0')
	plt.axis('equal')


def trilinear_covariance(n, var_diag, var_offdiag, device='cpu'):
	cm = torch.zeros(n, n).to(device)
	cm[torch.ones(n, n).diag().diag() == 1] = var_diag

	if var_offdiag:
		cm[torch.ones(n, n).tril(-1) == 1] = var_offdiag
		cm[torch.ones(n, n).triu(1) == 1] = var_offdiag

	return cm


def multivariate_prior(n, device='cpu', *args, **kwargs):
	# Same arguments as trilinear_covariance function
	cm = trilinear_covariance(n=n, device=device, *args, **kwargs)
	return MultivariateNormal(loc=torch.zeros(n).to(device), covariance_matrix=cm)


def p_to_prediction(p):

	if isinstance(p, list):
		return [p_to_prediction(_) for _ in p]

	if isinstance(p, Normal):
		pred = p.loc
	elif isinstance(p, Categorical):
		pred = p.logits.argmax(dim=1)
	elif isinstance(p, Bernoulli):
		pred = p.probs
	else:
		raise NotImplementedError

	return pred


__all__ = [
	'multiply_gaussians',
	'divide_gaussians',
	'cholesky_to_variance',
	'plot_covariance_ellipsoid',
	'trilinear_covariance',
	'multivariate_prior',
	'p_to_prediction',
]