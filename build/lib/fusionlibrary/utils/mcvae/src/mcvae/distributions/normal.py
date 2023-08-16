import matplotlib.pyplot as plt
from torch.distributions import Normal, kl_divergence, constraints
from torch.distributions.utils import _standard_normal, broadcast_all
from .utilities import multiply_gaussians, divide_gaussians
from .kl_utilities import KL_log_uniform


class Normal(Normal):

	arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
	support = constraints.real
	has_rsample = True
	_mean_carrier_measure = 0

	def __init__(
			self,
			loc,
			scale,
			scale_m=None,
			split_sizes=None,
			*args, **kwargs,
	):
		super().__init__(loc, scale, *args, **kwargs)

		self.scale_m = scale_m
		if scale_m is not None:
			self.loc, self.scale_m = broadcast_all(loc, scale_m)

		if split_sizes is None:
			try:
				self.split_sizes = (self.loc.shape[0],)
			except:
				pass
		else:
			assert sum(split_sizes) == self.loc.shape[0]
			self.split_sizes = split_sizes

	@property
	def stddev(self):
		if self.scale_m is None:
			return self.scale
		else:
			return self.variance.pow(0.5)

	@property
	def variance(self):
		if self.scale_m is None:
			return self.scale.pow(2)
		else:
			return self.scale.pow(2) + self.scale_m.pow(2)

	def __mul__(self, other):
		"""
		The Product of two Gaussian PDFs is a Scaled (un-normalized) Gaussian PDF.
		Here we provide only the proper (normalized) PDF.
		"""
		if other == 1:
			return self

		mean, var = multiply_gaussians(self.mean, self.variance, other.mean, other.variance)

		return Normal(loc=mean, scale=var.pow(0.5))

	def __truediv__(self, other):

		if other == 1:
			return self

		mean, var = divide_gaussians(self.mean, self.variance, other.mean, other.variance)

		return Normal(loc=mean, scale=var.pow(0.5))

	def __pow__(self, power, modulo=None):
		assert isinstance(power, int)
		assert power >= 0
		if power is 0:
			return 1
		if power is 1:
			return self
		else:
			p = self
			for i in range(1, power):
				p *= self
			return p

	def kl_divergence(self, other):
		return kl_divergence(Normal(loc=self.loc, scale=self.stddev), other)

	def kl_divergence_rev(self, other):
		return kl_divergence(other, self)

	def kl_divergence_symm(self, other):
		return 0.5 * (self.kl_divergence(other) + self.kl_divergence_rev(other))

	def kl_from_log_uniform(self):
		return KL_log_uniform(mu=self.loc, logvar=self.scale.pow(2).log())

	def plot(self):
		if len(self.loc.shape) > 1:
			self._plot_n()
		else:
			x = self.sample((1000,)).sort(0)[0]
			p = self.log_prob(x).exp()
			plt.plot(x.detach().numpy(), p.detach().numpy(), '.')

	def _plot_n(self):
		for l, s in zip(self.loc, self.stddev):
			Normal(loc=l, scale=s).plot()