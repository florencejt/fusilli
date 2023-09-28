import torch
from torch.distributions import Bernoulli, Categorical, MultivariateNormal, kl_divergence
from .utils import Utilities, DEVICE
from ..distributions import Normal, compute_kl, compute_ll, compute_logvar


class VAE(torch.nn.Module, Utilities):

	def __init__(
			self,
			data=None,
			lat_dim=1,
			n_feats=1,
			n_labels=None,  # for classification decoders
			beta=1.0,  # for beta-VAE (kl weight coefficient)
			sparse=False,
			log_alpha=None,
			noise_init_logvar=-3,
			noise_fixed=False,
			bias_enc=True,
			bias_dec=True,
			adam_lr=1e-3,
			model_name_dict=None,
	):
		super().__init__()

		if data is not None:
			assert isinstance(data, torch.Tensor)
			self.data = data
			self.n_feats = data.shape[1]
		else:
			self.n_feats = n_feats
		self.n_labels = n_labels
		self.beta = beta
		self.lat_dim = lat_dim
		self.sparse = sparse
		self.log_alpha = log_alpha
		self.noise_init_logvar = noise_init_logvar
		self.noise_fixed = noise_fixed
		self.bias_enc = bias_enc
		self.bias_dec = bias_dec

		self.init_encoder()
		self.init_decoder()

		self.adam_lr = adam_lr
		self.init_optimizer()

		self.model_name_dict = model_name_dict
		self.init_names()  # Model name is produced once everything is set up!

	def init_encoder(self):
		# Encoders: random initialization of weights
		bias = self.bias_enc
		self.W_mu = torch.nn.Linear(self.n_feats, self.lat_dim, bias=bias)
		if self.sparse:
			if self.log_alpha is None:
				self.log_alpha = torch.nn.Parameter(torch.Tensor(1, self.lat_dim))
				torch.nn.init.normal_(self.log_alpha, 0.0, 0.01)
		else:
			self.W_logvar = torch.nn.Linear(self.n_feats, self.lat_dim, bias=bias)

	def init_decoder(self):
		# Decoders: random initialization of weights
		bias = self.bias_dec
		self.W_out = torch.nn.Linear(self.lat_dim, self.n_feats, bias=bias)
		tmp_noise_par = torch.FloatTensor(1, self.n_feats).fill_(self.noise_init_logvar)
		if self.noise_fixed:
			self.W_out_logvar = torch.nn.Parameter(data=tmp_noise_par, requires_grad=False)
		else:
			self.W_out_logvar = torch.nn.Parameter(data=tmp_noise_par, requires_grad=True)

		del tmp_noise_par

	def encode(self, x):
		"""
		:param x: list of datasets (Obs x Feats)
		:return: list of encoded distributions (one list element per dataset)
		"""
		x = x.to(DEVICE)
		mu = self.W_mu(x)
		if not self.sparse:
			logvar = self.W_logvar(x)
		else:
			logvar = compute_logvar(mu, self.log_alpha)

		return Normal(loc=mu, scale=logvar.exp().pow(0.5))

	def decode(self, z):
		pi = Normal(
			loc=self.W_out(z),
			scale=self.W_out_logvar.exp().pow(0.5)
		)
		return pi

	def forward(self, x, y=None):
		q = self.encode(x)
		posterior = q

		if self.training:
			z = posterior.rsample()
		else:
			z = posterior.loc
		p = self.decode(z)

		fwd_ret = {
			'y': x if y is None else y,
			'posterior': posterior,
			'p': p,
		}

		return fwd_ret

	def compute_kl(self, posterior):

		return compute_kl(p1=posterior, p2=Normal(0, 1), sparse=self.sparse)

	def loss_function(self, fwd_ret):
		y = fwd_ret['y']
		posterior = fwd_ret['posterior']
		p = fwd_ret['p']

		kl = self.beta * self.compute_kl(posterior)
		ll = compute_ll(p, y)

		total = kl - ll

		losses = {
			'total': total,
			'kl': kl,
			'll': ll,
		}

		if self.training:
			self.save_loss(losses)
			return total
		else:
			return losses

	@staticmethod
	def p_to_prediction(p):

		if isinstance(p, list):
			return [VAE.p_to_prediction(_) for _ in p]

		if isinstance(p, Normal):
			pred = p.loc
		elif isinstance(p, Categorical):
			pred = p.logits.argmax(dim=1)
		elif isinstance(p, Bernoulli):
			pred = p.probs
		else:
			raise NotImplementedError

		return pred

	@staticmethod
	def p_to_expected_value(p):

		if isinstance(p, list):
			return [VAE.p_to_prediction(_) for _ in p]

		if isinstance(p, Normal):
			pred = p.loc
		elif isinstance(p, Categorical):
			pred = p.probs  #  expected value is not defined for a Cat distrib. Careful when using this method
		elif isinstance(p, Bernoulli):
			pred = p.probs
		else:
			raise NotImplementedError

		return pred

	def reconstruct(self, x, sample=False):
		with torch.no_grad():
			q = self.encode(x)
			posterior = q
			if sample:
				z = posterior.sample()
			else:
				z = posterior.loc
			p = self.decode(z)

		return self.p_to_prediction(p)

	def predict(self, *args, **kwargs):
		return self.reconstruct(*args, **kwargs)

	def generate(self):

		if self.sparse:
			raise NotImplementedError

		z = Normal(torch.zeros(1, self.lat_dim), 1).sample()
		p = self.decode(z)

		if isinstance(p, Normal):
			return p.loc
		elif isinstance(p, Categorical):
			return p.logits.argmax(dim=1)
		else:
			raise NotImplementedError

	def init_optimizer(self):
		self.optimizer = torch.optim.Adam(
			self.parameters(), lr=self.adam_lr
		)

	@property
	def dropout(self):
		if self.sparse:
			alpha = torch.exp(self.log_alpha.detach())
			return alpha / (alpha + 1)
		else:
			raise NotImplementedError

	def extra_repr(self) -> str:
		extrapars = ['W_out_logvar', 'log_alpha']
		ret = ''
		for ep in extrapars:
			if isinstance(getattr(self, ep), torch.nn.Parameter):
				ret += f'({ep}): Parameter(shape={getattr(self, ep).shape})\n'

		return ret.rstrip('\n')


class TwoLayersVAE(VAE):

	def __init__(
			self,
			act_fun=None,
			hidden_dim=None,
			*args, **kwargs,

	):
		self.hidden_dim = hidden_dim
		if act_fun is not None:
			self.act_fun = act_fun
		else:
			self.act_fun = torch.nn.LeakyReLU

		super().__init__(
			*args, **kwargs,
		)

	def init_encoder(self):
		if self.hidden_dim is None:
			self.hidden_dim = (self.n_feats + self.lat_dim) // 2

		self.W_mu = torch.nn.Sequential(
			torch.nn.Linear(self.n_feats, self.hidden_dim),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(self.hidden_dim, self.lat_dim),
		)
		if self.sparse:
			if self.log_alpha is None:
				self.log_alpha = torch.nn.Parameter(torch.FloatTensor(1, self.lat_dim).normal_(0, 0.01))
		else:
			self.W_logvar = torch.nn.Sequential(
				torch.nn.Linear(self.n_feats, self.hidden_dim),
				torch.nn.LeakyReLU(),
				torch.nn.Linear(self.hidden_dim, self.lat_dim),
			)

	def init_decoder(self):
		# Decoders: random initialization of weights
		self.W_out = torch.nn.Sequential(
			torch.nn.Linear(self.lat_dim, self.hidden_dim),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(self.hidden_dim, self.n_feats),
		)
		tmp_noise_par = torch.FloatTensor(1, self.n_feats).fill_(self.noise_init_logvar)
		if self.noise_fixed:
			self.W_out_logvar = torch.nn.Parameter(data=tmp_noise_par, requires_grad=False)
		else:
			self.W_out_logvar = torch.nn.Parameter(data=tmp_noise_par, requires_grad=True)

		del tmp_noise_par


class ThreeLayersVAE(VAE):

	def __init__(
			self,
			act_fun=None,
			hidden_dim=None,
			*args, **kwargs,

	):
		self.hidden_dim = hidden_dim
		if act_fun is not None:
			self.act_fun = act_fun
		else:
			self.act_fun = torch.nn.LeakyReLU

		super().__init__(
			*args, **kwargs,
		)

	def init_encoder(self):
		if self.hidden_dim is None:
			self.hidden_dim = (self.n_feats + self.lat_dim) // 2

		self.W_mu = torch.nn.Sequential(
			torch.nn.Linear(self.n_feats, self.hidden_dim),
			self.act_fun(),
			torch.nn.Linear(self.hidden_dim, self.hidden_dim),
			self.act_fun(),
			torch.nn.Linear(self.hidden_dim, self.lat_dim),
		)
		if self.sparse:
			if self.log_alpha is None:
				self.log_alpha = torch.nn.Parameter(torch.FloatTensor(1, self.lat_dim).normal_(0, 0.01))
		else:
			self.W_logvar = torch.nn.Sequential(
				torch.nn.Linear(self.n_feats, self.hidden_dim),
				self.act_fun(),
				torch.nn.Linear(self.hidden_dim, self.hidden_dim),
				self.act_fun(),
				torch.nn.Linear(self.hidden_dim, self.lat_dim),
			)

	def init_decoder(self):
		# Decoders: random initialization of weights
		self.W_out = torch.nn.Sequential(
			torch.nn.Linear(self.lat_dim, self.hidden_dim),
			self.act_fun(),
			torch.nn.Linear(self.hidden_dim, self.hidden_dim),
			self.act_fun(),
			torch.nn.Linear(self.hidden_dim, self.n_feats),
		)
		tmp_noise_par = torch.FloatTensor(1, self.n_feats).fill_(self.noise_init_logvar)
		if self.noise_fixed:
			self.W_out_logvar = torch.nn.Parameter(data=tmp_noise_par, requires_grad=False)
		else:
			self.W_out_logvar = torch.nn.Parameter(data=tmp_noise_par, requires_grad=True)

		del tmp_noise_par


class ConditionalDistributionNet(torch.nn.Module):

	def __init__(self, in_features, out_features):

		super().__init__()

		self.mu = torch.nn.Linear(in_features, out_features)
		self.logvar = torch.nn.Linear(in_features, out_features)

	def forward(self, x):

		loc = self.mu(x)
		scale = self.logvar(x).exp().pow(0.5)

		return Normal(loc=loc, scale=scale)


class MyVAE(torch.nn.Module):

	def __init__(self, n_feats, lat_dim, *args, **kwargs):

		super().__init__()

		self.encode = ConditionalDistributionNet(
			in_features=n_feats,
			out_features=lat_dim,
		)
		self.decode = ConditionalDistributionNet(
			in_features=lat_dim,
			out_features=n_feats,
		)

	def forward(self, x):

		q = self.encode(x)
		z = q.rsample()
		p = self.decode(z)

		return x, q, p

	def loss_function(self, x, q, p):

		kl = kl_divergence(q, Normal(0, 1)).sum(1).mean(0)
		ll = p.log_prob(x).sum(1).mean(0)
		total = kl - ll

		return total


__all__ = [
	'VAE',
	'TwoLayersVAE',
	'ThreeLayersVAE',
	'ConditionalDistributionNet',
	'MyVAE',
]