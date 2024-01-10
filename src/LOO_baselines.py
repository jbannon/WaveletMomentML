from sklearn.metrics import brier_score_loss, log_loss
import sys 
import os 
import argparse
import yaml 
from typing import Union, List, Dict, Tuple
import numpy as np
import pickle 
import networkx as nx 
import pandas as pd
import utils


from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut

from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from WMTransform import WaveletMomentTransform, DiffusionWMT, GeometricWMT
from collections import defaultdict
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, precision_recall_curve, accuracy_score

import tqdm 
from sklearn.dummy import DummyClassifier




def main(
	config:Dict
	)->None:
	

	"""
	Experimental Parameters
	"""
	
	drug: str
	rng_seed: int 
	data_dir: str
	
	pval_thresh: float 
	num_iters: int 
	tissues: List[str] 
	
	test_size: float
	num_genes:int

	
	""" 
	Model Parameters
	"""
	
	model: str
	preprocessing_steps: List[str]
	cv_criterion:str
	

	drug, tissues, rng_seed, data_dir, num_iters, train_percentage, feature_names, feature_file_name =\
	 	utils.unpack_parameters(config['EXPERIMENT_PARAMS'])
	


	models, cv_criterion = utils.unpack_parameters(config['MODEL_PARAMS'])
	
	rng = np.random.RandomState(seed = rng_seed)
	
	
	for tissue in tqdm.tqdm(tissues,total = len(tissues)):
				
		expr_file = utils.make_data_file_name(data_dir,drug,tissue,'expression')
		response_file = utils.make_data_file_name(data_dir,drug,tissue,'response')
		feature_file = utils.make_data_file_name(data_dir, drug,tissue, feature_file_name)
		
		expression_data = pd.read_csv(expr_file)
		response_data = pd.read_csv(response_file)
		feature_data= pd.read_csv(feature_file)
		
		
		feature_data = feature_data[['Run_ID']+feature_names]
		
		
		results = defaultdict(list)
		res_dir = "../results/LOO/{d}/{t}/".format(d=drug,t=tissue)
		os.makedirs(res_dir,exist_ok=True)
		for feature in tqdm.tqdm(feature_names + ['TARGET'],leave = False):

			if feature in feature_names:
				X = feature_data[feature].values.reshape(-1,1)
			elif feature == 'TARGET':
				target_genes = utils.fetch_drug_targets(drug)
				X = expression_data[target_genes]
				if X.shape[1]>1:
					X = np.log2(X.values + 1)
				else:
					X = np.log2(expression_data[target_genes].values + 1).reshape(-1,1)
			y = response_data['Response'].values
			

			for model in models:
				classifier, param_grid = utils.make_model_and_param_grid(model)							
				loo = LeaveOneOut()

				for i, (train_index, test_index) in tqdm.tqdm(enumerate(loo.split(X)),
					total = X.shape[0],leave=False):
				
					X_train, X_test = X[train_index,:], X[test_index,:]
					y_train, y_test = y[train_index], y[test_index]
					
					clf = GridSearchCV(classifier,param_grid,scoring = cv_criterion)
					clf.fit(X_train,y_train)
			
					
					

					pred_bins = clf.best_estimator_.predict(X_test)
					pred_probs = clf.best_estimator_.predict_proba(X_test)
					results['feature'].append(feature)
					results['model'].append(model)
					results['i'].append(i)
					results['tissue'].append(tissue)
					results['y_true'].append(y_test[0])
					results['pred_bin'].append(pred_bins[0])
					results['pred_prob'].append(pred_probs[:,1][0])

					
					
		df = pd.DataFrame(results)
		df.to_csv("{r}/baselines.csv".format(r=res_dir))
		





if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("-config",help="The config file for these experiments")
	args = parser.parse_args()



	with open(args.config) as file:
		config = yaml.safe_load(file)

	main(config)
	