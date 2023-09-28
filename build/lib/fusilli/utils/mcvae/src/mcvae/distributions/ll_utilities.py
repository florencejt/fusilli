from torch.distributions import Normal, Categorical, Bernoulli, MultivariateNormal


def compute_ll(p, x):
	"""
	:param p: Normal: p.loc.shape = (n_obs, n_feats)
	:param x:
	:return: log-likelihood compatible with the distribution p
	"""
	if isinstance(p, Normal):
		ll = p.log_prob(x).sum(1, keepdims=True)
	elif isinstance(p, Categorical):
		ll = p.log_prob(x.view(-1))
	elif isinstance(p, MultivariateNormal):
		ll = p.log_prob(x).unsqueeze(1)  # MultiVariate already sums over dimensions
	else:
		raise NotImplementedError

	return ll.mean(0)


__all__ = [
	'compute_ll',
]