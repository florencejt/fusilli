import pandas as pd
import torch
from ..os import execute_process


def nvidia_query():
	# 1. get the list of gpu and status
	gpu_query_columns = ('index', 'uuid', 'name', 'temperature.gpu',
						 'utilization.gpu', 'memory.used', 'memory.total')
	gpu_list = []

	smi_output = execute_process(
		r'nvidia-smi --query-gpu={query_cols} --format=csv,noheader,nounits'.format(
			query_cols=','.join(gpu_query_columns)
		))

	for line in smi_output.split('\n'):
		if not line: continue
		query_results = line.split(',')

		g = {col_name: col_value.strip() for (col_name, col_value) in zip(gpu_query_columns, query_results)}
		gpu_list.append(g)

	return pd.DataFrame(gpu_list).set_index('index')


if torch.cuda.is_available():
	GPUS = nvidia_query()
	print(GPUS)
	DEVICE_ID = int(GPUS.sort_values('utilization.gpu').index[0])
	DEVICE = torch.device(f'cuda:{DEVICE_ID}')
	del GPUS
else:
	DEVICE = torch.device('cpu')


def gpu_info():

	if torch.cuda.is_available():
		info = {}
		gpuquery = 'nvidia-smi -q -d Utilization | grep Gpu'
		memquery = 'nvidia-smi -q -d Memory | grep -A3 FB | grep '

		ret = os.popen(gpuquery).read()
		info['gpu'] = [int(s) for s in ret.split() if s.isdigit()]

		ret = os.popen(memquery+'Total').read()
		info['mtotal'] = [int(s) for s in ret.split() if s.isdigit()]

		ret = os.popen(memquery+'Used').read()
		info['mused'] = [int(s) for s in ret.split() if s.isdigit()]

		ret = os.popen(memquery+'Free').read()
		info['mfree'] = [int(s) for s in ret.split() if s.isdigit()]

		return info
	else:
		print('No GPU found!')


__all__ = [
	'DEVICE',
	'nvidia_query',
	'gpu_info',
]