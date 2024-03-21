# Borrowed from https://github.com/salesforce/ensemble-of-averages/blob/main/domainbed/EoA.py
# modified to fit SKD
import json
import numpy as np
from domainbed import datasets
from domainbed import algorithms
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed import networks
import torch
import torch.nn as nn
import os
import argparse
import time

# Copyright (c) Salesforce and its affiliates. All Rights Reserved
import json
import numpy as np
from domainbed import datasets
from domainbed import algorithms
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed import networks
import torch
import torch.nn as nn
import os
import argparse
import time

class Algorithm(torch.nn.Module):
	def __init__(self, input_shape, hparams, num_classes):
		super(Algorithm, self).__init__()
		self.hparams = hparams
		self.featurizer = networks.Student_Featurizer(input_shape, hparams)
		self.classifier = networks.Classifier(
			self.featurizer.n_outputs,
			num_classes,
			hparams['nonlinear_classifier'])

		self.network = nn.Sequential(self.featurizer, self.classifier)
		
		self.featurizer_mo = networks.Student_Featurizer(input_shape, hparams)
		self.classifier_mo = networks.Classifier(
			self.featurizer.n_outputs,
			num_classes,
			hparams['nonlinear_classifier'])
		
		self.network = self.network.cuda()
		self.network = torch.nn.parallel.DataParallel(self.network).cuda()

		self.network_sma = nn.Sequential(self.featurizer_mo, self.classifier_mo)
		self.network_sma = self.network_sma.cuda()
		self.network_sma = torch.nn.parallel.DataParallel(self.network_sma).cuda()
		
	def predict(self, x):
		if self.hparams['SMA']:
			return self.network_sma(x)
		else:
			return self.network(x)

def accuracy(models, loader):
	correct = 0
	total = 0
	weights_offset = 0
	with torch.no_grad():
		for data in loader:
			x1,y = data[0], data[-1]
			x = x1.cuda()
			y = y.cuda()
	
			p = None
			for model in models:
				model.eval()
				p_i = model.predict(x).detach()
				if p is None:
					p = p_i
				else:
					p += p_i
		   
			batch_weights = torch.ones(len(x))
		   
			batch_weights = batch_weights.cuda()
			if p.size(1) == 1:
				correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
			else:
				correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
			total += batch_weights.sum().item()
	return correct / total

#------------------------------------------------------------------------------------------------------------
def get_folders_for_dataset(path, dataset_name):
    # This function returns the subdirectories in the given path
    # that contain results for the specified dataset

    dataset_folders = []

    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)
        if os.path.isdir(subdir_path):
            results_path = os.path.join(subdir_path, "results.jsonl")
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    for line in f:
                        result = json.loads(line[:-1])
                        if 'dataset' in result['args'] and result['args']['dataset'] == dataset_name:
                            dataset_folders.append(subdir_path)
                            break

    return dataset_folders



def rename_dict(D):
	dnew = {}
	for key, val in D.items():
		pre = key.split('.')[0]
		if pre=='network':
			knew = '.'.join(['network.module'] + key.split('.')[1:])
		else:
			knew = '.'.join(['network_sma.module'] + key.split('.')[1:])
		dnew[knew] = val
	return dnew

def get_test_env_id(path):
	results_path = os.path.join(path, "results.jsonl")
	with open(results_path, "r") as f:
		for j,line in enumerate(f):
			r = json.loads(line[:-1])
			env_id = r['args']['test_envs'][0]
			break
	return env_id

def get_valid_model_selection_paths(path, nenv=4):
	valid_model_id = [[] for _ in range(nenv)]
	for env in range(nenv):
		cnt=0
		for i, subdir in enumerate(os.listdir(path)):
			if '.' not in subdir:
				test_env_id =get_test_env_id(os.path.join(path, subdir))
				if env==test_env_id:
					cnt+=1
					valid_model_id[env].append(f'{path}/{subdir}/best_model.pkl')
	return valid_model_id

def get_ensemble_test_acc(exp_path, nenv, dataset_name, data_dir, hparams, force=False, var=False, file_path=None, save_path = None):
	test_acc = {}
	save_path = args.save_dir + '/EoA' + '_' + args.dataset + '.txt'
	print("save_path: ", save_path)
	with open(save_path, 'w') as f:
		pass

	for env in range(nenv):
		dataset = vars(datasets)[dataset_name](data_dir, [env], hparams)
		assert nenv == len(dataset)
		test_acc[env] = None
		print(f'Test Domain: {dataset.ENVIRONMENTS[env]}')
		data_loader = FastDataLoader(
				dataset=dataset[env],
				batch_size=hparams['batch_size'],
				num_workers=hparams['num_workers']) 

		valid_model_id = get_valid_model_selection_paths(exp_path, nenv=len(dataset))
		Algorithm_all = []
		dataset_folders = get_folders_for_dataset(exp_path, dataset_name)
		for folder in dataset_folders:
			for model_path in valid_model_id[env]:
				if folder in model_path:
					Algorithm_ = Algorithm(dataset.input_shape, hparams, dataset.num_classes)
					algorithm_dict = torch.load(model_path)
					
					D = rename_dict(algorithm_dict['model_dict'])
					Algorithm_.load_state_dict(D, strict=False)
					Algorithm_all.append(Algorithm_)

		acc = accuracy(Algorithm_all, data_loader)
		print(f'  Test domain Acc: {100.*acc:.2f}%')
	
		with open(save_path, 'a') as f:
			f.write(f'Test Domain: {dataset.ENVIRONMENTS[env]}\n')
			f.write(f'Test domain Acc: {100.*acc:.2f}%\n')
			

		test_acc[env] = acc
	return test_acc





parser = argparse.ArgumentParser(description='Ensemble of Averages')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--dataset', type=str, default="PACS")
parser.add_argument('--resnet18', type=bool, default=False)
parser.add_argument('--output_dir', type=str, help='the experiment directory where the results of domainbed.scripts.sweep were saved')
parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
parser.add_argument('--save_dir', type=str, help='directory to save the result file')
args = parser.parse_args()
dataset_name= args.dataset
if dataset_name in ['PACS', 'TerraIncognita', 'VLCS', 'OfficeHome']:
	nenv = 4
elif dataset_name=='DomainNet':
	nenv = 6

data_dir= args.data_dir
hparams = {'data_augmentation': False, "nonlinear_classifier": False, "resnet_dropout": 0, "resnet18": False, "batch_size": 128, "num_workers":8, "SMA": True, 'student_model':'resnet'}
if args.hparams:
	hparams.update(json.loads(args.hparams))

path = args.output_dir

tic = time.time()
test_acc = get_ensemble_test_acc(path, nenv, dataset_name, data_dir, hparams, force=False, save_path = args.save_dir)
test_acc = {k: float(f'{100.*test_acc[k]:.1f}') for k in test_acc.keys()}
toc = time.time()
print(f'Avg: {np.array(list(test_acc.values())).mean():.1f}, Time taken: {toc-tic:.2f}s')

save_path = args.save_dir + '/EoA' + '_' + args.dataset + '.txt'
with open(save_path, 'a') as f:
	f.write(f'Avg: {np.array(list(test_acc.values())).mean():.1f}, Time taken: {toc-tic:.2f}s')

