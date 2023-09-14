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
	
	gex_units, genesets, models, rng_seed, split_type, geneset_base, base_path, n_splits, n_trials = io_utils.unpack_parameters(config['PARAMETERS'])

	rng = np.random.RandomState(seed = rng_seed)


	if split_type == 'stratified':
		splitter = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = rng)

	elif split_type == 'montecarlo':
		splitter = None

	elif split_type ==  'loo':
		splitter = LeaveOneOut()

	cv_score = 'roc_auc'
	

	for drug in ['Atezo', 'Ipi','Nivo', 'Pembro']:

		results = pd.DataFrame()

		for tissue in utils.DRUG_TISSUE_MAP[drug]:
			dataSet = io_utils.load_ICIDataSet(base_path,drug, tissue, gex_units)
		
			X = dataSet.X 
			y = dataSet.y
			

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
						if geneset == 'target':
							genes = [utils.TARGET_GENE_MAP[utils.DRUG_TARGET_MAP[drug]]]
						else:
							genes = geneset_utils.load_geneset(geneset_base,geneset)


						common_features = list(set(genes).intersection(set(dataSet.genes_to_idx.keys())))
						feature_string = geneset


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
		
		print(results.head())
		results.to_csv("../results/{d}_baselines.csv")	
				



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("-config",help="The config file for these experiments")
	args = parser.parse_args()
	


	with open(args.config) as file:
		config = yaml.safe_load(file)
	
	main(config)
	
			


