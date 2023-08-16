import torch
from torch.distributions import kl_divergence


def compute_log_alpha(mu, logvar):
	# clamp because dropout rate p in 0-99%, where p = alpha/(alpha+1)
	return (logvar - 2 * torch.log(torch.abs(mu) + 1e-8)).clamp(min=-8, max=8)


def compute_logvar(mu, log_alpha):
	return log_alpha + 2 * torch.log(torch.abs(mu) + 1e-8)


def compute_clip_mask(mu, logvar, thresh=3):
	# thresh < 3 means p < 0.95
	# clamp dropout rate p in 0-99%, where p = alpha/(alpha+1)
	log_alpha = compute_log_alpha(mu, logvar)
	return (log_alpha < thresh).float()


def KL_log_uniform(mu, logvar):
	"""
	Paragraph 4.2 from:
	Variational Dropout Sparsifies Deep Neural Networks
	Molchanov, Dmitry; Ashukha, Arsenii; Vetrov, Dmitry
	https://arxiv.org/abs/1701.05369
	https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/blob/master/KL%20approximation.ipynb
	"""
	log_alpha = compute_log_alpha(mu, logvar)
	k1, k2, k3 = 0.63576, 1.8732, 1.48695
	neg_KL = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) - k1
	return -neg_KL


def compute_kl(p1, p2=None, sparse=False):
	"""
	:param p1: Normal distribution with p1.loc.shape = (n_obs, n_lat_dims)
	:param p2: same as p1
	:param sparse:
	:return: scalar value
	"""
	if sparse:
		kl = KL_log_uniform(mu=p1.loc, logvar=p1.scale.pow(2).log())
	else:
		kl = kl_divergence(p1, p2)

	return kl.sum(1, keepdims=True).mean(0)


__all__ = [
	'compute_log_alpha',
	'compute_logvar',
	'compute_clip_mask',
	'KL_log_uniform',
	'compute_kl',
]