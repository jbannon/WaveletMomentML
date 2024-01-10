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


from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from WMTransform import WaveletMomentTransform, DiffusionWMT, GeometricWMT
from collections import defaultdict
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, precision_recall_curve, accuracy_score

import tqdm 
from sklearn.dummy import DummyClassifier
import math



def main(
	config:Dict
	)->None:
	

	"""
	Experimental Parameters
	"""
	
	drug: str
	rng_seed: int 
	data_dir: str
	count_cutoff:int 
	pval_thresh: float 
	num_iters: int 
	tissues: List[str]
	
	""" 
	Model Parameters
	"""
	model: str
	preprocessingSteps: List[str]
	cvCriterion:str

	
	

	drug, tissues, rng_seed, data_dir, num_iters, train_percentage, feature_names, feature_file_name =\
	 	utils.unpack_parameters(config['EXPERIMENT_PARAMS'])
	


	models, cv_criterion = utils.unpack_parameters(config['MODEL_PARAMS'])
	



	min_size, max_size, step, round_value = utils.unpack_parameters(config['PAUCITY_PARAMS'])

	rng = np.random.RandomState(seed = rng_seed)
	
	
	
	train_sizes = np.round(np.arange(min_size,max_size,step),round_value)


	for tissue in tqdm.tqdm(tissues,total = len(tissues)):
			
		expr_file = utils.make_data_file_name(data_dir,drug,tissue,'expression')
		response_file = utils.make_data_file_name(data_dir,drug,tissue,'response')
		feature_file = utils.make_data_file_name(data_dir, drug,tissue, feature_file_name)
		
		expression_data = pd.read_csv(expr_file)
		response_data = pd.read_csv(response_file)
		feature_data= pd.read_csv(feature_file)
		
		
		feature_data = feature_data[['Run_ID']+feature_names]
		
		results = defaultdict(list)
		res_dir = "../results/paucity/{d}/{t}/".format(d=drug,t=tissue)
		
		os.makedirs(res_dir,exist_ok=True)

		for feature in tqdm.tqdm(feature_names + ["TARGET"], leave = False):
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
			

			
			for model in tqdm.tqdm(models):
				classifier, param_grid = utils.make_model_and_param_grid(model)							
					
				for train_size in tqdm.tqdm(train_sizes):
					# print(trainSize)
					for i in tqdm.tqdm(range(num_iters),leave = False):
						X_train, X_test, y_train, y_test =\
							train_test_split(X,y,train_size = train_size, stratify=y)
					
		
						try:
							clf = GridSearchCV(classifier,param_grid,scoring = cv_criterion, verbose = 0)
							clf.fit(X_train,y_train)
						except Exception as error:
							print(error)
							print(step)
							print(feature)
							print(np.isnan(X))
							sys.exit(1)
				
						pred_bins = clf.predict(X_test)
						pred_probs = clf.predict_proba(X_test)
					

						# binary results
						acc = accuracy_score(y_test,pred_bins)
						tn, fp, fn, tp = confusion_matrix(y_test, pred_bins,labels = [0,1]).ravel()
				
						try:
							roc_auc = roc_auc_score(y_test,pred_probs[:,1])
						except:
							print(y_train)
							print(np.amax(X))
							sys.exit(1)
				
						brier = brier_score_loss(y_test, pred_probs[:,1])
						logloss = log_loss(y_test,pred_probs[:,1])
						

						if model == "LR":
							k = X_train.shape[1]
							n = X_train.shape[0]
							train_pred_probs = clf.predict_proba(X_train)
							LL = log_loss(y_train,train_pred_probs[:,1],)
							BIC = k*math.log(n) - 2*LL
						else:
							BIC = "NA"

					
						results['iter'].append(i)
						results['drug'].append(drug)
						results['tissue'].append(tissue)
						results['model'].append(model)
						results['feature'].append(feature)
						results['BIC'].append(BIC)
						results['train_size'].append(train_size)
						results['acc'].append(acc)
						results['tp'].append(tp)
						results['tn'].append(tn)
						results['fp'].append(fp)
						results['fn'].append(fn)		
						results['roc_auc'].append(roc_auc)
						results['brier'].append(brier)
						results['log_loss'].append(logloss)	

					
						
		df = pd.DataFrame(results)
		df.to_csv("{r}/baselines.csv".format(r= res_dir))
		




if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("-config",help="The config file for these experiments")
	args = parser.parse_args()



	with open(args.config) as file:
		config = yaml.safe_load(file)

	main(config)
	