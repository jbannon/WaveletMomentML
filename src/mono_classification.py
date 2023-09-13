import sys 
import os 
import argparse
import yaml 
from typing import Union, List, Dict, Tuple
import io_utils, utils, model_utils, geneset_utils, graph_utils
import numpy as np
import pickle 
import networkx as nx 
import pandas as pd

from sklearn.model_selection import LeaveOneOut,GridSearchCV, KFold, StratifiedKFold

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from WMTransform import WaveletMomentTransform, DiffusionWMT
from collections import defaultdict

from sklearn.metrics import confusion_matrix, roc_auc_score, auc,  average_precision_score, f1_score
from sklearn.metrics import precision_recall_curve, accuracy_score, balanced_accuracy_score
import tqdm 

from sklearn.dummy import DummyClassifier




def main(
	config:Dict
	)->None:

	
	drug, gex_units, base_path = io_utils.unpack_parameters(config['DATA_PARAMS'])
	
	rng_seed, n_splits, shuffle, models, cv_score, res_path, split_type, n_trials = io_utils.unpack_parameters(config['EXPERIMENT_PARAMS'])
	
	geneset_base, genesets, target_cutoff  = io_utils.unpack_parameters(config['GENESET_PARAMS'])

	genesets = genesets + [utils.DRUG_TARGET_MAP[drug]]
	

	net_basepath, network_types, weightings = io_utils.unpack_parameters(config['NETWORK_PARAMS'])


	rng = np.random.RandomState(seed = rng_seed)


	split_type = split_type.lower()
	network_file = "{base}/{type}/{weight}.pickle".format(base = net_basepath,type = network_types[0], weight = weightings[0])
	
	with open(network_file, 'rb') as istream:
		G = pickle.load(istream)

	tissues = utils.DRUG_TISSUE_MAP[drug]
	
	
	if split_type == 'stratified':
		splitter = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = rng)

	elif split_type == 'montecarlo':
		splitter = None

	elif split_type ==  'loo':
		splitter = LeaveOneOut()
	



	for tissue in tissues:

		# include studies??

		dataSet = io_utils.load_ICIDataSet(base_path,drug, tissue, gex_units)
		
		results = pd.DataFrame()
		X = dataSet.X 
		y = dataSet.y
		results = pd.DataFrame()

		

		for model in models: 
			
			if model.lower() in ['uniform','stratified','most_frequent']:

				classifier = DummyClassifier(strategy = model)
				param_grid = {}
				X_ = X.copy()

				if split_type == 'stratified':

					model_results = model_utils.run_kfold_trials(X_,y,drug,tissue, model,'all genes',
						classifier, param_grid,splitter = splitter,cv_score = cv_score)

				elif split_type == 'montecarlo':

					model_results = model_utils.run_monte_carlo_trials(n_trials, X_,y,drug,tissue, model,'all genes',
						classifier, param_grid,rng = rng,cv_score = cv_score)

				elif split_type == 'loo':
					# print(splitter)
					model_results = model_utils.run_LOO_trials(X_,y,drug,tissue, model,'all genes',
						classifier, param_grid,splitter = splitter,cv_score = cv_score,idx_to_id =dataSet.idx_to_id)
		
				results = pd.concat((results,model_results),axis=0)	
				


			else:
				standardize_features = True
				prefix = 'clf'

				classifier, param_grid = model_utils.make_model_and_param_grid(model, 
					 standardize_features,
					 prefix, 					
					 rng)
				
				
				for geneset in genesets:

					print("geneset: {g}".format(g=geneset))
					
					if geneset.lower()!='target':
						
						if geneset.upper() in ['PD-L1','PD1','CTLA4']:
							genes = geneset_utils.load_netprop_geneset(geneset_base,geneset,target_cutoff)
						else:

							geneset_string = geneset
							genes = geneset_utils.load_geneset(geneset_base,geneset)

					
						common_features = list(set(genes).intersection(set(dataSet.genes_to_idx.keys())))

						LCC_graph = graph_utils.harmonize_graph_and_geneset(G,common_features)
						if len(LCC_graph.nodes())<10:
							geneset_dict = {'common':common_features}
							
						else:
							geneset_dict = {'common':common_features,
										'LCC':[v for v in LCC_graph.nodes()],
										'wm':[v for v in LCC_graph.nodes()]
									}
					
					
					
						for k in geneset_dict.keys():
							feature_string = "{g}_{k}".format(g=geneset,k=k)
							print("{k}:\t {c} genes".format(k= k, c =len(geneset_dict[k])))
							
						
							X_ = X[:,[dataSet.genes_to_idx[v] for v in geneset_dict[k]]]

							if k == 'wm':
								transformer = DiffusionWMT(2,3,nx.adjacency_matrix(LCC_graph).todense(),'diffu',True)
								X_ = transformer.computeTransform(X_)

							if split_type == 'stratified':
								model_results = model_utils.run_kfold_trials(X_,y, drug, tissue, model, feature_string, 
									classifier, param_grid, splitter,cv_score)

							elif split_type == 'montecarlo':
								
								model_results = model_utils.run_monte_carlo_trials(n_trials, X_,y,drug,tissue, model,feature_string,
									classifier, param_grid,rng = rng,cv_score = cv_score)

							elif split_type == 'loo':

								model_results = model_utils.run_LOO_trials(X_,y,drug,tissue, model,feature_string,
									classifier, param_grid,splitter = splitter,cv_score = cv_score,idx_to_id = dataSet.idx_to_id)


							results = pd.concat((results,model_results),axis=0)

				
					else:
						
						genes = [utils.TARGET_GENE_MAP[utils.DRUG_TARGET_MAP[drug]]]
						
						X_ = X[:,[dataSet.genes_to_idx[v] for v in genes]]
						
						
						if split_type == 'stratified':
							model_results = model_utils.run_kfold_trials(X_,y, drug, tissue, model, "target_only", 
									classifier, param_grid, splitter,cv_score)

						elif split_type == 'montecarlo':
								
							model_results = model_utils.run_monte_carlo_trials(n_trials, X_,y,drug,tissue, model,"target_only",
									classifier, param_grid,rng = rng,cv_score = cv_score)

						elif split_type == 'loo':

							model_results = model_utils.run_LOO_trials(X_,y,drug,tissue, model,"target_only",
								classifier, param_grid,splitter = splitter,cv_score = cv_score,idx_to_id = dataSet.idx_to_id)

						results = pd.concat((results,model_results),axis=0)

					
						
						
		results = pd.DataFrame(results)
		print(results)
		results.to_csv("./dink.csv")
		res_base = "../results/mono_classification/{s}/{gex}/{d}/{t}/".format(s=split_type,gex = gex_units,d = drug, t=tissue)
		result_file_name = res_base + "results.csv"
		os.makedirs(res_base, exist_ok=True)
		results.to_csv(result_file_name)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("-config",help="The config file for these experiments")
	args = parser.parse_args()
	


	with open(args.config) as file:
		config = yaml.safe_load(file)
	
	main(config)