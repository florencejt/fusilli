#!/usr/bin/env python
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from mcvae.datasets import SyntheticDataset
from mcvae.models.mcvae import MtMcvae
from mcvae.utilities import preprocess_and_add_noise, simulate_mar_multi_channel_data
from mcvae.models.utils import DEVICE, load_or_fit, model_press
import argparse
import datetime


print(f"Start: {datetime.datetime.now()}")
print(f"Running on {DEVICE}")

parser = argparse.ArgumentParser(
	description='Synthetic experiment with missing data in MultiTask MCVAE modeling'
)
parser.add_argument(
	'--lat_dim',
	type=int,
	default=2,
	metavar='<int>',
	help='latent dimension'
)
parser.add_argument(
	'--n_feats',
	type=int,
	default=100,
	metavar='<int>',
	help='number of features per channel'
)
parser.add_argument(
	'--n_channels',
	type=int,
	default=3,
	metavar='<int>',
	help='n_channels'
)
parser.add_argument(
	'--n',
	type=int,
	default=500,
	metavar='<int>',
	help='n of observations per channel'
)
parser.add_argument(
	'--snr',
	type=float,
	default=10.0,
	metavar='<float>',
	help='snr level'
)
parser.add_argument(
	'--epochs',
	type=int,
	default=10000,
	metavar='<int>',
	help='training epochs'
)
parser.add_argument(
	'--fcd',
	type=float,
	default=0.75,
	metavar='<float>',
	help='fcd fraction of complete data'
)
parser.add_argument(
	'--model_seed',
	type=int,
	default=0,
	metavar='<int>',
	help='model_seed'
)
args = parser.parse_args()

lat_dim = args.lat_dim
n_feats = args.n_feats
n_channels = args.n_channels
n = args.n
snr = args.snr
epochs = args.epochs
fcd = args.fcd
model_seed = args.model_seed

stats = []

for fcd in (0.0, 0.25, 0.5, 0.75, 1.0):
	###########################
	## PREPARE TRAINING DATA ##
	###########################
	ds_train = SyntheticDataset(n=n, lat_dim=lat_dim, n_feats=n_feats, n_channels=n_channels, train=True)
	x_gt, x_noisy = preprocess_and_add_noise(ds_train.x, snr=snr)

	x_m, ids = simulate_mar_multi_channel_data(x=x_noisy, intersection_fraction=fcd)

	inters = np.zeros((len(ids), len(ids)))
	for i, idi in enumerate(ids):
		for j, idj in enumerate(ids):
			inters[i, j] = len(set(idi).intersection(set(idj)))

	unions = np.zeros((len(ids), len(ids)))
	for i, idi in enumerate(ids):
		for j, idj in enumerate(ids):
			unions[i, j] = len(set(idi).union(set(idj)))

	###################
	## MODEL FITTING ##
	###################

	# MCVAE
	torch.manual_seed(model_seed)
	model = MtMcvae(ids=[list(_) for _ in ids], lat_dim=lat_dim, data=x_m, sparse=False, noise_init_logvar=3)
	model.optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
	model.to(DEVICE)

	model_dir = Path('MODELS', 'Synthetic_MAR')
	model_dir.mkdir(parents=True, exist_ok=True) if not model_dir.exists() else None

	ptfile = Path(model_dir, f'mt-mcvae_fcd-{fcd}.pt')

	load_or_fit(model, model.data, epochs, ptfile, force_fit=False)

	#############
	## TESTING ##
	#############
	n_test = 1000
	ds_test = SyntheticDataset(n=n_test, lat_dim=lat_dim, n_feats=n_feats, n_channels=n_channels, train=False)
	x_gt_test, x_noisy_test = preprocess_and_add_noise(ds_test.x, snr=snr)
	x_rec_test = model.reconstruct(x_noisy_test)

	# PRESS = Predictive (test set) Sum of Squares
	test_press = sum(model_press(model, x_noisy_test, x_gt_test))

	tmp_stats = {
		'lat_dim': lat_dim,
		'n_feats': n_feats,
		'n_channels': n_channels,
		'n': n,
		'snr': f'{snr:.1f}',
		'epochs': epochs,
		'fcd': fcd,
		'model_seed': model_seed,
		'test_press': float(f'{test_press:.2f}'),
	}

	stats.append(tmp_stats)
	del tmp_stats

df = pd.DataFrame(stats)
df.plot(x='fcd', y='test_press', logy=True)

print(f"End: {datetime.datetime.now()}\nSee you!")
