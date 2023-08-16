import copy
from operator import add
from functools import reduce
import numpy as np


def ids_to_selection(ids):
	# TODO: this code is not very readable. Try to improve it.
	"""
	Return a list of list with none in Channel_i=Channel_o positions and common ids between Channel_i and Channel_o
	otherwise.
	:param id: list list of ids: example [list1, list2, list3]
	:return: example [[None, dict, dict], [dict, None, dict], [dict, dict, None]]
	"""
	for i in ids:
		assert sorted(i) == i
	sel = []
	for i in range(len(ids)):  # for every input channel
		sel.append([])
		for o in range(len(ids)):  # for every output channel
			# inner join: common elements between input and output channel
			inner_join = np.sort(tuple(set(ids[i]) & set(ids[o])))
			if i is o or len(inner_join) is 0:
				# If Input and output are the same then there is no need to compute the JOIN SET
				# Likewise if the join set is empty
				sel[i].append(None)
			else:
				sel[i].append({})
				# index of the elements of the input channel which are in common with the output channel
				input_index = copy.deepcopy([ids[i].index(element) for element in inner_join])
				sel[i][o]['input'] = input_index
				# index of the elements of the output channel which are in common with the input channel
				output_index = copy.deepcopy([ids[o].index(element) for element in inner_join])
				sel[i][o]['output'] = output_index
				del output_index, input_index
			del inner_join
	return sel


def process_ids(ids):
	"""
	Depends on "ids_to_selection" function.
	:param ids: list of list of ids. len(ids) = N channels
	:return: indeces to compute appropriately KL and Log-Likelihood
	"""
	if isinstance(ids, tuple):
		return process_ids([list(_) for _ in ids])

	assert isinstance(ids, list)
	for l in ids:
		assert sorted(l) == l

	union = sorted(set(reduce(add, ids)))
	intersection = sorted(reduce(set.intersection, [set(_) for _ in ids]))
	z_index_list = []
	for l in ids:
		# for every channel, find the index of elements for which to compute the KL
		z_index_list.append([union.index(i) for i in l])

	return {
		'union': union, 'intersection': intersection,
		'z_index': z_index_list, 'LL_index': ids_to_selection(ids)
	}


def negate_ids(ids):
	# return index of nan position
	assert isinstance(ids, tuple)
	union = sorted(set(reduce(add, ids)))
	nids = tuple([tuple([s for s in union if s not in l]) for l in ids])
	return nids


def mark_missing_as_none(x, ids, default_value=np.nan):
	assert isinstance(x, list)
	feats = [_.shape[1] for _ in x]
	union = sorted(set(reduce(add, ids)))
	X = []
	for ch in range(len(x)):
		X.append([])
		for s in union:
			if s in ids[ch]:
				iloc = ids[ch].index(s)
				X[ch].append(x[ch][iloc].tolist())
			else:
				X[ch].append((default_value * np.ones(feats[ch])).tolist())
	return [np.array(_) for _ in X]


def mse_gt_vs_imputed(x_gt, x_im, ids):
	"""
	:param x_gt: ground truth
	:param x_im: imputed data
	:param ids: index of missing features
	"""
	n = len(x_gt)
	for _ in [x_gt, x_im, ids]:
		assert isinstance(_, list) or isinstance(_, tuple)
		assert len(_) is n

	x_gt = [np.array(_.tolist()) for _ in x_gt]
	x_im = [np.array(_.tolist()) for _ in x_im]

	mse = 0
	nids = negate_ids(ids)
	for i, xgt, xim in zip(nids, x_gt, x_im):
		mse += np.mean((xgt[i, ] - xim[i, ]) ** 2)

	mse /= n
	return mse


__all__ = [
	'ids_to_selection',
	'process_ids',
	'negate_ids',
	'mark_missing_as_none',
	'mse_gt_vs_imputed',
]