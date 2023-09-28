import copy
import datetime
import socket
import numpy as np
import torch
from torch.nn.utils.convert_parameters import _check_param_device
from functools import reduce
from pathlib import Path
from ..gpu import DEVICE


class Running():
	def __init__(self, runningpath):
		self.runningpath = Path(runningpath)

	def __enter__(self):
		print(f'\tCreating {self.runningpath}')
		self.creation_time = datetime.datetime.now()
		self.runningpath.touch()
		with open(self.runningpath, 'a') as f:
			f.write(f'{self.creation_time}\n')
			f.write(f'{socket.gethostname()}\n')
		print(f"\tCreated: {self.creation_time}")

	def __exit__(self, exc_type, exc_val, exc_tb):
		print(f'\tDeleting {self.runningpath}')
		self.deletion_time = datetime.datetime.now()
		try:
			self.runningpath.unlink()
			print(f"\tDeleted: {self.deletion_time}")
		except FileNotFoundError:
			print(f"\tAnother process already deleted the file.")
		print(f"\t\tElapsed: {self.deletion_time - self.creation_time}")


def update_partial_state_dict(model, state_dict):
	# useful when parameters in model and state_dict not all are coincident
	# e.g. when importing some layers from already trained models.
	sd = model.state_dict()
	sd.update(
		{k: v for k, v in state_dict.items() if k in sd}
	)
	model.load_state_dict(sd)


def update_model_attr_from_dict(model, dictionary):

	for attr, value in dictionary.items():
		if attr == 'state_dict':
			try:
				model.load_state_dict(value)
			except RuntimeError:
				print('\tWarning: not all parameters have been updated.')
				update_partial_state_dict(model, value)
		elif attr == 'optimizer':
			pass
			# model.optimizer.load_state_dict(value)
		elif attr == 'scheduler':
			pass
			# model.scheduler.load_state_dict(value)
		else:
			setattr(model, attr, value)


def update_model(model, modelpath, device=DEVICE, verbose=False):
	print(f"Loading {modelpath}") if verbose else None
	mdict = torch.load(modelpath, map_location=device)
	update_model_attr_from_dict(model, mdict)
	model.eval()
	del mdict


def save_model(model, modelpath, verbose=False, warn=False):
	"""
	Adapted from:
	https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/3
	http://archive.is/2pnMw
	"""
	try:
		loss = model.loss['total']
		if str(loss[-1]) == 'nan':
			raise ValueError("Loss is nan! Not saving.")
		# elif loss[-1] > loss[0]:
		# 	raise ValueError(f"Loss diverged! Not saving.\n\tloss[-1]={loss[-1]}\n\t\nloss[0]={loss[0]}")
	except AttributeError:
		pass

	state = {
		'state_dict': model.state_dict(),
		'optimizer': model.optimizer.state_dict(),
	}
	try:
		state['scheduler'] = model.scheduler.state_dict()
	except AttributeError:
		warnings.warn(f"No 'scheduler' to save!") if warn else None

	for attr in [
		'loss',
		'minibatch_loss',
		'epochs',
		'journal',
		'global_distribution',
		'path_teacher',  # for EP trained (student) models
		'cavity',
		'data',
		'ids',  # case with missing data
	]:
		try:
			state[attr] = getattr(model, attr)
		except AttributeError:
			warnings.warn(f"No '{attr}' to save!") if warn else None

	print(f"Saving on {modelpath}") if verbose else None
	state['path'] = modelpath
	torch.save(state, modelpath)


def load_or_fit(model, data, epochs, ptfile, init_loss=True, minibatch=False, force_fit=False):

	"""
	Routine to train or load a model.
	:param model: model to optimize.
	:param data: training data. It can be also a PyTorch DataLoader for a dataset that do not fit in memory.
	:param epochs: number of training epochs.
	:param ptfile: path to *.pt file where to save the trained model.
	:param minibatch: True if training with mini-batches.
	:param force_fit: force the training even if the model is already trained.
	"""

	ptfile = Path(ptfile)

	rf = Path(f'{ptfile}.running')

	if ptfile.exists():
		try:
			rf.unlink()
		except FileNotFoundError:
			pass

	if ptfile.exists() and not force_fit:
		update_model(model, ptfile, verbose=True)
	elif not rf.exists():
		with Running(rf):
			start_at = datetime.datetime.now()
			print(f"Start fitting: {start_at}")
			print(f'\tModel destination: {ptfile}')
			if init_loss:
				model.init_loss(minibatch=minibatch)
			model.optimize(epochs=epochs, data=data)
			save_model(model, ptfile)
			end_at = datetime.datetime.now()
			print(f"End fitting: {end_at}")
			print(f"\tElapsed: {end_at - start_at}")
	elif rf.exists():
		print('Someone else is taking care of fitting this model.')


# TESTING
def classification_accuracy(y, y_pred):
	if isinstance(y, list):
		return [classification_accuracy(i, j) for i, j in zip(y, y_pred)]
	total_correct = (y == y_pred).sum().item()
	accuracy = 100 * total_correct / len(y)
	return accuracy


def cv_classification_accuracy(y, y_pred, n=1000):
	if isinstance(y, list):
		return [cv_classification_accuracy(i, j, n=n) for i, j in zip(y, y_pred)]
	N = len(y)
	assert n <= N
	assert len(y) == len(y_pred)
	splits = N // n
	i = 0
	acc = []
	for _ in range(splits):
		acc.append(classification_accuracy(y[i:i + n], y_pred[i:i + n]))
		i += n
	return np.mean(acc), np.std(acc)


def press(a, b):
	if isinstance(a, list):
		ssqe = [press(a[i], b[i]) for i in range(len(a))]
	else:
		ssqe = ((a - b)**2).sum().item()
	return ssqe


def model_press(model, x, y):
	"""
	predicted error sum of squares
	"""
	y_pred = model.reconstruct(x)
	return press(y_pred, y)


# gradient stop
def stop_grad_in_mcreg(model):
	for c in range(model.n_channels):
		if c not in model.enc_channels:
			model.vae[c].W_mu.requires_grad_(False)
			if not model.sparse:
				model.vae[c].W_logvar.requires_grad_(False)
		if c not in model.dec_channels:
			model.vae[c].W_out.requires_grad_(False)
			model.vae[c].W_out_logvar.requires_grad_(False)


def common_observations(obs_list, id_list):
	"""
	missing to full for mcvae vs mcvaeWmissing
	"""

	assert isinstance(obs_list, list)
	assert isinstance(id_list, list)
	assert len(obs_list) == len(id_list)
	for _, __ in zip(obs_list, id_list):
		assert len(_) == len(__)

	common_ids = reduce(lambda s1, s2: set(s1).intersection(s2), id_list)

	ret_list = []
	for _obs, _ids in zip(obs_list, id_list):
		sel = [_ids.index(_) for _ in common_ids]
		ret_list.append(_obs[sel])

	return ret_list


def init_names(classname, namedict=None):
	model_name = classname
	if not namedict == None:
		for key in sorted(namedict):
			val = namedict[key]
			if type(val) == list or type(val) == tuple:
				# val = '_'.join([str(i) for i in val])
				val = str(np.sum(val))
			model_name += '__' + key + '_' + str(val)
	return model_name


def load_data_from_spreadsheet(filepath):
	"""
	Utility to load multi-channel observations from a spreadsheet.
	The spreadsheet should contain one sheet per channel.
	The observation identifier is assumed to be in the first column of every sheet.
	:param filepath: path to spreadsheet file.
	:return: X, ids
	"""
	pass


class Utilities:

	def get_parameters(self):
		return parameters_to_vector(self.parameters()).tolist()

	def get_grad(self):
		param_device = None

		vec = []
		for param in self.parameters():
			# Ensure the parameters are located in the same device
			param_device = _check_param_device(param, param_device)

			vec.append(param.grad.view(-1))
			grad_vec = torch.cat(vec)
			norm_grad = grad_vec.norm(2)
		return {
			'grad': grad_vec.tolist(),
			'norm': norm_grad.item(),
		}

	def optimize_batch(self, local_batch):

		fwd_return = self.forward(local_batch)
		loss = self.loss_function(fwd_return)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.detach().item()

	def init_optimization(self):
		self.train()  # Inherited method which sets self.training = True

	def optimize(self, epochs, data):

		self.init_optimization()

		try:
			self.epochs.append(self.epochs[-1]+epochs)
		except AttributeError:
			self.epochs = [0, epochs]

		self.minibatch = hasattr(self, 'minibatch_loss')
		self.n_batches = len(data) if self.minibatch else 1
		logging_interval = 1 if self.minibatch else 10

		for epoch in range(self.epochs[-2], self.epochs[-1]):
			if self.minibatch:
				counter = 0
				for local_batch in data:
					loss = self.optimize_batch(local_batch)
					counter += 1
					# if counter % 10 == 0:
					# 	print(f"Batch #{counter}/{self.n_batches - 1}", end='\t')
				self.save_minibatch_loss()
			else:
				loss = self.optimize_batch(data)

			try:
				self.scheduler.step(loss)  # For "ReduceLROnPlateau" scheduler
			except AttributeError:
				pass

			#try:
				# scheduler for the learning rate.
				# See for example "StepLR" in https://pytorch.org/docs/stable/optim.html
				# self.scheduler.step()
				# self.scheduler.step(self.loss['total'][-1])  # For "ReduceLROnPlateau" scheduler
			#except AttributeError:
			#	pass

			# self.save_journal(epoch)

			if np.isnan(loss):
				print('Loss is nan!\nTry to reduce the learning rate')
				self.eval()
				break

			# if self.optimizer.state_dict()['param_groups'][0]['lr'] < 1e-5:
			# 	print(f'Learning rate < {1e-5}. Stop optimization.')
			# 	self.end_optimization()
			# 	break

			if epoch % logging_interval == 0:
				self.print_loss(epoch) if not self.minibatch else self.print_minibatch_loss(epoch)
				# self.save_log(epoch)
				# if loss_has_diverged(self.loss['total']):
				# 	print('Loss diverged!')
				# 	break

		self.end_optimization()

	def end_optimization(self):
		self.eval()  # Inherited method which sets self.training = False
		self.epochs[-1] = len(self.loss['total']) if not self.minibatch else len(self.minibatch_loss['total'])
		self.eval()

	def init_loss(self, minibatch=False):
		empty_loss = {
			'total': [],
			'kl': [],
			'll': [],
		}
		self.loss = copy.deepcopy(empty_loss)

		if minibatch:
			self.minibatch_loss = copy.deepcopy(empty_loss)

	def print_loss(self, epoch):
		total = self.loss['total'][-1]
		ll = self.loss['ll'][-1]
		kl = self.loss['kl'][-1]
		print('====> Epoch: {:4d}/{} ({:.0f}%)\tLoss: {:.4f}\tLL: {:.4f}\tKL: {:.4f}\tLL/KL: {:.4f}'.format(
			epoch,
			self.epochs[-1],
			100. * (epoch) / self.epochs[-1],
			total, ll, kl, ll / (1e-8 + kl),
		), end='\n')

	def print_minibatch_loss(self, epoch):
		total = self.minibatch_loss['total'][-1]
		ll = self.minibatch_loss['ll'][-1]
		kl = self.minibatch_loss['kl'][-1]
		print('====> Epoch: {:4d}/{} ({:.0f}%)\tLoss: {:.4f}\tLL: {:.4f}\tKL: {:.4f}\tLL/KL: {:.4f}'.format(
			epoch,
			self.epochs[-1],
			100. * (epoch) / self.epochs[-1],
			total, ll, kl, ll / (1e-8 + kl),
		), end='\n')

	def save_loss(self, losses):
		for key in losses.keys():
			self.loss[key].append(float(losses[key].detach().item()))

	def save_minibatch_loss(self):
		for k, v in self.loss.items():
			new_v = float(sum(v[-self.n_batches:]) / self.n_batches)
			self.minibatch_loss[k].append(new_v)

	def init_names(self):
		self.model_name = init_names(self._get_name(), self.model_name_dict)


__all__ = [
	'DEVICE',
	'update_partial_state_dict',
	'update_model_attr_from_dict',
	'update_model',
	'save_model',
	'load_or_fit',
	# TESTING
	'classification_accuracy',
	'cv_classification_accuracy',
	'press',
	'model_press',
	# Stop Gradient
	'stop_grad_in_mcreg',
	'common_observations',
	# Optimization
	'Utilities',
]
