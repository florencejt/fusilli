import numpy as np
import torch
from torch.utils.data import Dataset


class GeneratorInt(torch.nn.Module):

	def __init__(
			self,
			lat_dim=2,
			n_channels=2,
			n_feats=5,
			seed=100,
	):
		super().__init__()

		self.lat_dim = lat_dim
		self.n_channels = n_channels
		self.n_feats = n_feats

		#  Save random state (http://archive.is/9glXO)
		np.random.seed(seed)  # or None
		# self.random_state = np.random.get_state()  # get the initial state of the RNG

		W = []

		for ch in range(n_channels):
			w = np.random.randint(-8, 8, size=(self.n_feats, self.lat_dim))
			W.append(torch.nn.Linear(self.lat_dim, self.n_feats, bias=False))
			W[ch].weight.data = torch.FloatTensor(w)

		self.W = torch.nn.ModuleList(W)

	def forward(self, z):
		if type(z) == np.ndarray:
			z = torch.FloatTensor(z)

		assert z.size(1) == self.lat_dim

		obs = []
		for ch in range(self.n_channels):
			x = self.W[ch](z)
			obs.append(x.detach())

		return obs


class GeneratorUniform(torch.nn.Module):

	def __init__(
			self,
			lat_dim=2,
			n_channels=2,
			n_feats=5,
			seed=100,
	):
		"""
		Generate multiple sources (channels) of data through a linear generative model:

		z ~ N(0,I)

		for ch in N_channels:
			x_ch = W_ch(z)

		where:

			"W_ch" is an arbitrary linear mapping z -> x_ch

		:param lat_dim:
		:param n_channels:
		:param n_feats:
		"""
		super().__init__()

		self.lat_dim = lat_dim
		self.n_channels = n_channels
		self.n_feats = n_feats

		self.seed = seed
		np.random.seed(self.seed)

		W = []

		for ch in range(n_channels):
			w_ = np.random.uniform(-1, 1, (self.n_feats, lat_dim))
			u, s, vt = np.linalg.svd(w_, full_matrices=False)
			w = u if self.n_feats >= lat_dim else vt
			W.append(torch.nn.Linear(lat_dim, self.n_feats, bias=False))
			W[ch].weight.data = torch.FloatTensor(w)

		self.W = torch.nn.ModuleList(W)

	def forward(self, z):

		if isinstance(z, list):
			return [self.forward(_) for _ in z]

		if type(z) == np.ndarray:
			z = torch.FloatTensor(z)

		assert z.size(1) == self.lat_dim

		obs = []
		for ch in range(self.n_channels):
			x = self.W[ch](z)
			obs.append(x.detach())

		return obs


class SyntheticDataset(Dataset):

	def __init__(
			self,
			n=500,
			lat_dim=2,
			n_feats=5,
			n_channels=2,
			generatorclass=GeneratorUniform,
			train=True,
	):
		self.n = n  # N subjects
		self.lat_dim = lat_dim
		self.n_feats = n_feats
		self.n_channels = n_channels
		self.train = train
		seed = 7 if self.train is True else 14
		np.random.seed(seed)
		self.z = np.random.normal(size=(self.n, self.lat_dim))

		self.generator = generatorclass(lat_dim=self.lat_dim, n_channels=self.n_channels, n_feats=self.n_feats)

		self.x = self.generator(self.z)

	def __len__(self):

		return self.n

	def __getitem__(self, item):

		return [x[item] for x in self.x]
