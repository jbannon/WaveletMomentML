import sys 
import numpy as np
from typing import Union, List, Dict, Tuple
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA
from collections import defaultdict
import tqdm 

from sklearn.metrics import confusion_matrix, roc_auc_score, auc,  average_precision_score, f1_score
from sklearn.metrics import precision_recall_curve, accuracy_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut

def make_model_and_param_grid(
	model_name:str,
	standardize:bool,
	prefix:str,
	rng):
	
	if model_name == 'SVC':
		model = (prefix, LinearSVC())
		param_grid = {'dual':['auto'],'C':np.arange(0.1, 1, 0.1)}
	elif model_name == 'KNN':
		model = (prefix, KNeighborsClassifier())
		param_grid = {'n_neighbors':np.arange(2,10), 'weights':['uniform','distance']}
	elif model_name == 'RBF':
		model = (prefix, SVC(kernel = 'rbf'))
		param_grid = {'gamma':['scale','auto'],'C':np.arange(0.1, 1, 0.1), 'class_weight':['balanced'] }
	elif model_name == 'LR':
		model = (prefix, LogisticRegression())
		param_grid = {'penalty':['l2'], 'max_iter':[10**10], 'solver':['lbfgs'], 'C':np.arange(0.1, 1, 0.1), 'class_weight':['balanced'] }

	if standardize:
		model = [('scaler',StandardScaler())]+[model]

	param_grid = {prefix+"__"+k:param_grid[k] for k in param_grid}

	return Pipeline(model), param_grid




def run_kfold_trials(
	X:np.ndarray, 
	y:np.ndarray,
	drug:str,
	tissue:str, 
	model:str,
	feature:str,
	classifier, 
	param_grid,
	splitter, 
	cv_score:str):
	

	
	results = defaultdict(list)
	
	for i, (train_idx, test_idx) in tqdm.tqdm(enumerate(splitter.split(X,y)),
						total = splitter.get_n_splits(),
						desc='k-fold split'):

		X_train, X_test, y_train, y_test = X[train_idx,:], X[test_idx,:], y[train_idx], y[test_idx]


		clf = GridSearchCV(classifier,param_grid,scoring=cv_score)

		clf.fit(X_train,y_train)

		pred_bins = clf.best_estimator_.predict(X_test)

		acc = accuracy_score(y_test,pred_bins)
		bal_acc = balanced_accuracy_score(y_test,pred_bins)


		tn, fp, fn, tp = confusion_matrix(y_test, pred_bins,labels = [0,1]).ravel()
		
		f1 = f1_score(y_test,pred_bins)


		results['fold'].append(i)
		results['drug'].append(drug)
		results['tissue'].append(tissue)
		results['model'].append(model)
		results['feature'].append(feature)
		results['acc'].append(acc)
		results['bal_acc'].append(bal_acc)
		results['tp'].append(tp)
		results['tn'].append(tn)
		results['fp'].append(fp)
		results['fn'].append(fn)
		results['f1'].append(f1)


		if model in ['RFC','GBC','LR']:

			pred_probs = clf.best_estimator_.predict_proba(X_test)
			av_prec = average_precision_score(y_test, pred_probs[:,1])
			roc_auc = roc_auc_score(y_test,pred_probs[:,1])
			precision, recall, threshs = precision_recall_curve(y_test, pred_probs[:,1])

			pr_auc = auc(recall,precision)

			results['av_prec'].append(av_prec)
			results['roc_auc'].append(roc_auc)
			results['pr_auc'].append(pr_auc)
		else:
			results['av_prec'].append(-1)
			results['roc_auc'].append(-1)
			results['pr_auc'].append(-1)




	df = pd.DataFrame(results)
	return df


def run_monte_carlo_trials(
	n_trials:int,
	X:np.ndarray, 
	y:np.ndarray,
	drug:str,
	tissue:str, 
	model:str,
	feature:str,
	classifier, 
	param_grid,
	rng, 
	cv_score:str) -> pd.DataFrame:


	results = defaultdict(list)


	for i in tqdm.tqdm(range(n_trials),desc=' monte carlo trials'):

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = rng,stratify = y)

		clf = GridSearchCV(classifier,param_grid,scoring=cv_score)

		clf.fit(X_train,y_train)

		pred_bins = clf.best_estimator_.predict(X_test)

		acc = accuracy_score(y_test,pred_bins)
		bal_acc = balanced_accuracy_score(y_test,pred_bins)


		tn, fp, fn, tp = confusion_matrix(y_test, pred_bins,labels = [0,1]).ravel()
		
		f1 = f1_score(y_test,pred_bins)

		
		results['round'].append(rep)
		results['iter'].append(i)
		results['drug'].append(drug)
		results['tissue'].append(tissue)
		results['model'].append(model)
		results['feature'].append(feature)
		results['acc'].append(acc)
		results['bal_acc'].append(bal_acc)
		results['tp'].append(tp)
		results['tn'].append(tn)
		results['fp'].append(fp)
		results['fn'].append(fn)
		results['f1'].append(f1)


		if model in ['RFC','GBC','LR']:

			pred_probs = clf.best_estimator_.predict_proba(X_test)
			av_prec = average_precision_score(y_test, pred_probs[:,1])
			roc_auc = roc_auc_score(y_test,pred_probs[:,1])
			precision, recall, threshs = precision_recall_curve(y_test, pred_probs[:,1])

			pr_auc = auc(recall,precision)

			results['av_prec'].append(av_prec)
			results['roc_auc'].append(roc_auc)
			results['pr_auc'].append(pr_auc)
		else:
			results['av_prec'].append(-1)
			results['roc_auc'].append(-1)
			results['pr_auc'].append(-1)




	df = pd.DataFrame(results)
	return df



def run_LOO_trials(
	X:np.ndarray, 
	y:np.ndarray,
	drug:str,
	tissue:str, 
	model:str,
	feature:str,
	classifier, 
	param_grid,
	splitter,
	cv_score:str,
	idx_to_id:Dict[int,str] 
	) -> pd.DataFrame:


	results = defaultdict(list)


	for i, (train_idx, test_idx) in tqdm.tqdm(enumerate(splitter.split(X)),total = X.shape[0]):

		X_train, X_test, y_train, y_test = X[train_idx,:], X[test_idx,:], y[train_idx], y[test_idx]
		
		clf = GridSearchCV(classifier,param_grid,scoring=cv_score)

		clf.fit(X_train,y_train)

		y_pred = clf.best_estimator_.predict(X_test)[0]

		results['id'].append(idx_to_id[i])
		results['model'].append(model)
		results['drug'].append(drug)
		results['tissue'].append(tissue)
		results['feature'].append(feature)
		results['true y'].append(y_test[0])
		results['pred_y'].append(y_pred)


		if model in ['stratified','most_frequent','uniform','RFC','GBC','LR']:
			# print(clf.best_estimator_.predict_proba(X_test))
			pred_probs = clf.best_estimator_.predict_proba(X_test)[0][1]
			
		else:
			pred_probs = -1

		results['pred_prob'].append(pred_probs)


	df = pd.DataFrame(results)
	return df
