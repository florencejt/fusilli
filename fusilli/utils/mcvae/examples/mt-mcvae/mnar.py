#!/usr/bin/env python
from pathlib import Path
import torch
from mcvae.datasets import SyntheticDataset
from mcvae.models.mcvae import MtMcvae
from mcvae.utilities import preprocess_and_add_noise, simulate_mnar_multi_channel_data
from mcvae.models.utils import DEVICE, load_or_fit, model_press
from mcvae.imputation import *
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
	default=8,
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
	default=5,
	metavar='<int>',
	help='n_channels'
)
parser.add_argument(
	'--n',
	type=int,
	default=1000,
	metavar='<int>',
	help='n of observations per channel'
)
parser.add_argument(
	'--snr',
	type=float,
	default=100.0,
	metavar='<float>',
	help='snr level'
)
parser.add_argument(
	'--lambda_g',
	type=int,
	default=0,
	metavar='<int>',
	help='lambda g (global)'
)
parser.add_argument(
	'--lambda_p',
	type=int,
	default=3,
	metavar='<int>',
	help='lambda p (pairwise)'
)
parser.add_argument(
	'--epochs',
	type=int,
	default=10000,
	metavar='<int>',
	help='training epochs'
)
parser.add_argument(
	'--model_seed',
	type=int,
	default=0,
	metavar='<int>',
	help='model_seed'
)
args = parser.parse_args()
for k, v in args._get_kwargs():
	exec(f'{k} = {v}')

###########################
## PREPARE TRAINING DATA ##
###########################
ds_train = SyntheticDataset(n=n, lat_dim=lat_dim, n_feats=n_feats, n_channels=n_channels, train=True)
x_gt, x_noisy = preprocess_and_add_noise(ds_train.x, snr=snr)

x_m, ids = simulate_mnar_multi_channel_data(x=x_noisy, n_datasets=n_channels, lambda_p=lambda_p, lambda_g=lambda_g)

###################
## MODEL FITTING ##
###################
torch.manual_seed(model_seed)
model = MtMcvae(ids=ids, lat_dim=lat_dim, data=x_m, sparse=False)
model.optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
model.to(DEVICE)

# model0_dir = Path('MODELS', 'Synthetic_MNAR', f'MissMcvae_e_{epochs}_nc_{n_channels}_nf_{n_feats}_ld_{lat_dim}_n_{n}_snr_{snr}_lg_{lambda_g}_lp_{lambda_p}_ms_{0}')
model_dir = Path('MODELS', 'Synthetic_MNAR', f'{model._get_name()}_e_{epochs}_nc_{n_channels}_nf_{n_feats}_ld_{lat_dim}_n_{n}_snr_{snr}_lg_{lambda_g}_lp_{lambda_p}_ms_{model_seed}')
model_dir.mkdir(parents=True, exist_ok=True) if not model_dir.exists() else None

modelbasename = f'{model_dir}/model'
ptfile = f'{modelbasename}.pt'

load_or_fit(model, model.data, epochs, ptfile)

###########################
## DENOISING AUTOENCODER ##
###########################
# x_da = mark_missing_as_none(x_m, ids)
# sizes_da = [_.shape[1] for _ in x_da]
# X_da = torch.cat([torch.Tensor(_) for _ in x_da], dim=1)
#
# torch.manual_seed(model_seed)
# model_da = DenoisingAutoencoder(data=X_da)
# model_da.optimizer = torch.optim.SGD(model_da.parameters(), momentum=0.99, lr=0.01, nesterov=True)
# model_da.to(DEVICE)
#
# modelbasename_da = f'{model_dir}/model_da'
# ptfile_da = Path(f'{modelbasename_da}.pt')
#
# load_or_fit(model_da, model_da.data, epochs, ptfile_da)

#####################
## TEST IMPUTATION ##
#####################
x_im_mcvae = model.impute(x_m, ids)
mse_mcvae = mse_gt_vs_imputed(x_gt, x_im_mcvae, ids)

# X_im_da = model_da(model_da.data)['out'].detach().to('cpu')
# x_im_da = [_.numpy() for _ in torch.split_with_sizes(X_im_da, sizes_da, dim=1)]
# mse_da = mse_gt_vs_imputed(x_gt, x_im_da, ids)

# x_im_knn1 = multi_channel_knn_impute(x_m, ids, k=1)
# mse_knn1 = mse_gt_vs_imputed(x_gt, x_im_knn1, ids)
# x_im_knn3 = multi_channel_knn_impute(x_m, ids, k=3)
# mse_knn3 = mse_gt_vs_imputed(x_gt, x_im_knn3, ids)
# x_im_knn5 = multi_channel_knn_impute(x_m, ids, k=5)
# mse_knn5 = mse_gt_vs_imputed(x_gt, x_im_knn5, ids)

# itimp_file = Path(f'{model0_dir}/iterative_imputer.pt')
# if itimp_file.exists():
# 	# do not seed this imputer: use model0_dir as saving directory
# 	itimp_file = Path(f'{model0_dir}/iterative_imputer.pt')
# 	if itimp_file.exists():
# 		print("Loading Iterative Imputer")
# 		itimp = torch.load(itimp_file)
# 		x_im_itimp, itimp = multi_channel_iterative_impute(x_m, ids, imputer=itimp)
# 	else:
# 		x_im_itimp, itimp = multi_channel_iterative_impute(x_m, ids)
# 		torch.save(itimp, itimp_file)
# 	del itimp  # this is a huge file. Get rid of it as soon as you can.
#
# 	mse_itimp = mse_gt_vs_imputed(x_gt, x_im_itimp, ids)
# else:
# 	mse_itimp = np.nan

stats = {
	'lat_dim': lat_dim,
	'n_feats': n_feats,
	'n_channels': n_channels,
	'n': n,
	'snr': f'{snr:.1f}',
	'epochs': epochs,
	'model_seed': model_seed,
	'lambda_g': lambda_g,
	'lambda_p': lambda_p,
	# 'mse_mcvae': f'{mse_mcvae:.4}',
	# 'mse_knn1': f'{mse_knn1:.4}',
	# 'mse_knn3': f'{mse_knn3:.4}',
	# 'mse_knn5': f'{mse_knn5:.4}',
	# 'mse_itimp': f'{mse_itimp:.4}',
	# 'mse_dae': f'{mse_da:.4}',
}

#############
## TESTING ##
#############
n_test = 1000
ds_test = SyntheticDataset(n=n_test, lat_dim=lat_dim, n_feats=n_feats, n_channels=n_channels, train=False)
x_gt_test, x_noisy_test = preprocess_and_add_noise(ds_test.x, snr=snr)
# x_m_test, ids_test = simulate_mnar_multi_channel_data(x=x_noisy_test, n_datasets=n_channels, lambda_p=lambda_p, lambda_g=lambda_g)

x_rec_test = model.reconstruct(x_noisy_test)
test_press = sum(model_press(model, x_noisy_test, x_gt_test))

stats['n_test'] = n_test
stats['test_press'] = f'{test_press:.5}'

print(f'{stats}\n')

print(f"End: {datetime.datetime.now()}\nSee you!")
