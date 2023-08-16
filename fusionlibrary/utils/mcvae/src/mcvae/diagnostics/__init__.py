import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse
import itertools


def plot_dropout(model, sort=True):
	# Show dropout effect
	do = model.dropout.detach().numpy().reshape(-1)
	if sort:
		do = np.sort(do)
	plt.figure()
	plt.bar(range(len(do)), do)
	plt.suptitle(f'Dropout probability of {model.lat_dim} fitted latent dimensions in Sparse Model')


def plot_latent_space(model, data=None, qzx=None, classificator=None, text=None, uncertainty=True, comp=None, fig_path=None):
	if data is None:
		try:
			data = model.data
		except:
			pass

	channels = model.n_channels
	if not hasattr(model, 'ch_name'):
		model.ch_name = [f'Ch.{_}' for _ in range(channels)]

	comps = model.lat_dim
	# output = model(data)
	# qzx = output['qzx']
	if qzx is None:
		qzx = model.encode(data)

	if classificator is not None:
		groups = np.unique(classificator)
		if not groups.dtype == np.dtype('O'):
			# remove nans if groups are not objects (strings)
			groups = groups[~np.isnan(groups)]

	# One figure per latent component
	#  Linear relationships expected between channels
	if comp is not None:
		itercomps = comp if isinstance(comp, list) else [comp]
	else:
		itercomps = range(comps)

	# One figure per channel
	#  Uncorrelated relationsips expected between latent components
	for comp in itercomps:
		fig, axs = plt.subplots(channels, channels)
		fig.suptitle(r'$z_{' + str(comp) + '}$', fontsize=30)
		for i, j in itertools.product(range(channels), range(channels)):
			if i == j:
				axs[j, i].text(
					0.5, 0.5, f'z|{model.ch_name[i]}',
					horizontalalignment='center', verticalalignment='center',
					fontsize=20
				)
				axs[j, i].axis('off')
			elif i > j:
				xi = qzx[i].loc.detach().numpy()[:, comp]
				xj = qzx[j].loc.detach().numpy()[:, comp]
				si = np.exp(0.5*qzx[i].scale.detach().numpy()[:, comp])
				sj = np.exp(0.5 * qzx[j].scale.detach().numpy()[:, comp])
				ells = [Ellipse(xy=[xi[p], xj[p]], width=2 * si[p], height=2 * sj[p]) for p in range(len(xi))]
				if classificator is not None:
					for g in groups:
						g_idx = classificator == g
						axs[j, i].plot(xi[g_idx], xj[g_idx], '.', alpha=0.5, markersize=15)
						if uncertainty:
							color = ax.get_lines()[-1].get_color()
							for idx in np.where(g_idx)[0]:
								axs[j, i].add_artist(ells[idx])
								ells[idx].set_alpha(0.1)
								ells[idx].set_facecolor(color)
				else:
					axs[j, i].plot(xi, xj, '.')
					if uncertainty:
						for e in ells:
							axs[j, i].add_artist(e)
							e.set_alpha(0.1)
				if text is not None:
					[axs[j, i].text(*item) for item in zip(xi, xj, text)]
				# Bisettrice
				lox, hix = axs[j, i].get_xlim()
				loy, hiy = axs[j, i].get_ylim()
				lo, hi = np.min([lox, loy]), np.max([hix, hiy])
				axs[j, i].plot([lo, hi], [lo, hi], ls="--", c=".3")
			else:
				axs[j, i].axis('off')

		if classificator is not None:
			[axs[-1, 0].plot(0,0) for g in groups]
			legend = ['{} (n={})'.format(g, len(classificator[classificator==g])) for g in groups]
			axs[-1,0].legend(legend)
			try:
				axs[-1, 0].set_title(classificator.name)
			except AttributeError:
				axs[-1, 0].set_title('Groups')

	if fig_path is not None:
		pickle.dump(fig, open(fig_path + '.pkl', 'wb'))
		fs = 2 * channels + 1
		plt.rcParams['figure.figsize'] = (fs, fs)
		# plt.tight_layout()
		plt.savefig(fig_path, bbox_inches='tight')
		plt.close()

	return fig, axs


def plot_loss(model, stop_at_convergence=True, fig_path=None, skip=0):
	true_epochs = len(model.loss['total']) - 1
	if skip	> 0:
		print(f'skipping first {skip} epochs where losses might be very high')
	losses = np.array([model.loss[key][skip:] for key in model.loss.keys()]).T
	fig = plt.figure()
	try:
		plt.suptitle('Model ' + str(model.model_name) + '\n')
	except:
		pass
	plt.subplot(1, 2, 1)
	plt.title('Loss (common scale)')
	plt.xlabel('epoch')
	plt.plot(losses), plt.legend(model.loss.keys())
	if not stop_at_convergence:
		plt.xlim([0, model.epochs])
	plt.subplot(1, 2, 2)
	plt.title('loss (relative scale)')
	plt.xlabel('epoch')
	max_losses = 1e-8 + np.max(np.abs(losses), axis=0)
	plt.plot(losses / max_losses), plt.legend(model.loss.keys())
	if not stop_at_convergence:
		plt.xlim([0, model.epochs])

	if fig_path is not None:
		# pickle.dump(fig, open(fig_path + '.pkl', 'wb'))
		plt.rcParams['figure.figsize'] = (8, 5)
		plt.savefig(f'{fig_path}.png', bbox_inches='tight')
		plt.close()


def plot_weights(model, side='decoder', title = '', xlabel='', comp=None, rotation=15, fig_path=None):
	try:
		model.n_channels = model.n_datasets
	except:
		pass
	if not hasattr(model, 'ch_name'):
		model.ch_name = [f'Ch.{_}' for _ in range(model.n_channels)]
	fig, axs = plt.subplots(model.n_channels, 1)
	if comp is None:
		suptitle = 'Model Weights\n({})'.format(side)
	else:
		suptitle = 'Model Weights\n({}, comp. {})'.format(side, comp)
	plt.suptitle(title + suptitle)
	for ch in range(model.n_channels):
		ax = axs if model.n_channels == 1 else axs[ch]  # 'AxesSubplot' object does not support indexing
		x = np.arange(model.n_feats[ch])
		if side == 'encoder':
			y = model.vae[ch].W_mu.weight.detach().numpy().T.copy()
		else:
			y = model.vae[ch].W_out.weight.detach().numpy().copy()
			try:  # In case of bernoulli features
				if model.bern_feats is not None:
					bidx = model.bern_feats[ch]
					y[bidx, :] = sigmoid(y[bidx, :])
			except:
				pass
		if y.shape[0] > 200:
			pass
		else:
			if comp is not None:
				y = y[:, comp]
			# axs[ch].plot(y)
			if model.lat_dim == 1 or comp is not None:
				ax.bar(x, y.reshape(-1), width=0.25)
			else:
				ax.plot(x, y)
			ax.set_ylabel(model.ch_name[ch], rotation=45, fontsize=14)
			if comp is None:
				ax.legend(['comp. '+str(c) for c in range(model.lat_dim)])
			ax.axhline(y=0, ls="--", c=".3")
			try:
				tick_marks = np.arange(len(model.varname[ch]))
				ax.set_xticks(tick_marks)
				ax.set_xticklabels(model.varname[ch], rotation=rotation, fontsize=20)
			except:
				pass
			plt.xlabel(xlabel)

	if fig_path is not None:
		pickle.dump(fig, open(fig_path+'.pkl', 'wb'))
		fsch = 4 * model.n_channels
		fsfeat = 3 * max(model.n_feats)
		plt.rcParams['figure.figsize'] = (fsfeat, fsch)
		plt.tight_layout()
		plt.savefig(fig_path, bbox_inches='tight')
		plt.close()


__all__ = [
	'plot_dropout',
	'plot_latent_space',
	'plot_loss',
	'plot_weights',
]