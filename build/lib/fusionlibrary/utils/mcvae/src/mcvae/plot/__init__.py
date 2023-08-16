import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import itertools


def lsplom(X, ax_ref=None, title='', classificator=None, names=None, figsize=None):
	"""
	inspired by list "splom" function in R.
	"""
	assert type(X) == list

	if ax_ref is None:
		ax_ref = X
	if names is None:
		names = ['Ch.{}'.format(i) for i in range(len(X))]

	def fix_axes(fig, ax_ref):
		min = np.min([x.min() for x in ax_ref])
		max = np.max([x.max() for x in ax_ref])
		for i, ax in enumerate(fig.axes):
			# ax.text(0, 0, "ax%d" % (i+1), va="center", ha="center")
			ax.set_xlim([min, max])
			ax.set_ylim([min, max])
			# Hide the right and top spines
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.spines['left'].set_visible(False)
			ax.spines['bottom'].set_visible(False)
			# Only show ticks on the left and bottom spines
			ax.yaxis.set_ticks_position('left')
			ax.xaxis.set_ticks_position('bottom')
			for tl in ax.get_xticklabels() + ax.get_yticklabels():
				# tl.set_visible(False)
				pass

	N = len(X)
	f = plt.figure(figsize=figsize)
	plt.suptitle(title, fontsize=18)
	for n, x in enumerate(X):
		features = x.shape[1]
		gs = GridSpec(features, features)
		gs.update(left=n / N + 0.05, right=(n + 1) / N, wspace=0.05)
		for i, j in itertools.product(range(features), range(features)):
			plt.subplot(gs[i, j])
			if i > j:
				if classificator is None:
					plt.plot(x[:, j], x[:, i], '.')
				else:
					groups = np.unique(classificator[n])
					groups = groups[~np.isnan(groups)]
					for g in groups:
						g_idx = classificator[n] == g
						plt.plot(x[g_idx, j], x[g_idx, i], '.', alpha=0.55, markersize=10)
				if False:
					plt.plot(x[0:2, j], x[0:2, i], 'r-', markersize=2)
					plt.plot(x[1:3, j], x[1:3, i], 'g-', markersize=2)
					plt.plot(x[2:4, j], x[2:4, i], 'k-', markersize=2)
				# zero axis
				plt.axhline(y=0, ls="--", c=".3")
				plt.axvline(x=0, ls="--", c=".3")
			elif i == j:
				if i == 0:
					plt.title(names[n], fontsize=22)
				plt.text(
					0, 0, 'feat' + str(i),
					horizontalalignment='center', verticalalignment='center',
					fontsize=18
				)
				plt.axis('off')
			else:
				plt.axis('off')
	fix_axes(f, ax_ref)