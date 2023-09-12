import sys 
import os 
import argparse
import yaml 
from typing import Union, List, Dict, Tuple

import numpy as np
import pickle 
import networkx as nx 
import pandas as pd

from sklearn.model_selection import LeaveOneOut,GridSearchCV, KFold, StratifiedKFold

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from WMTransform import DiffusionWMT
from collections import defaultdict

from sklearn.metrics import confusion_matrix, roc_auc_score, auc,  average_precision_score, f1_score
from sklearn.metrics import precision_recall_curve, accuracy_score, balanced_accuracy_score
import tqdm 

from sklearn.preprocessing import FunctionTransformer
import io_utils, utils, model_utils, geneset_utils, graph_utils
from sklearn.dummy import DummyClassifier




def main():

	MAX_SCALES = 6
	MAX_MOMENTS = 6
	network_path_base = "../data/networks/cri"
	genesets = ['auslander','LINCS']
	geneset_base = "../data/genesets/"
	base_path = "../data/expression/cri"
	gex_units = 'log_tpm'
	target_cutoff = 200
	num_folds = 5
	splitter = StratifiedKFold(n_splits = num_folds,shuffle = True,random_state = 1234)
	


	for drug in utils.DRUG_TISSUE_MAP.keys():
		results = defaultdict(list)
		

		for sparsity in ['sparse','dense']:
		
			for weighting in ['weighted','unweighted']:
			
				network_file = "{b}/{s}/{w}.pickle".format(b=network_path_base,s = sparsity,w = weighting)

			
				with open(network_file, 'rb') as istream:
					G = pickle.load(istream)
			
			
				

				for tissue in utils.DRUG_TISSUE_MAP[drug]:
					
					dataSet = io_utils.load_ICIDataSet(base_path, drug, tissue, gex_units)
					genesets = genesets + [utils.DRUG_TARGET_MAP[drug]]
					X = dataSet.X
					y = dataSet.y
							
					for geneset in genesets:
						if geneset.upper() in ['PD-L1','PD1','CTLA4']:
						
							geneset_string = 'target'
							genes = geneset_utils.load_netprop_geneset(geneset_base,geneset,target_cutoff)
					
						else:
						
							geneset_string = geneset
							genes = geneset_utils.load_geneset(geneset_base, geneset)
						
						
						common_features = list(set(genes).intersection(set(dataSet.genes_to_idx.keys())))
						LCC_graph = graph_utils.harmonize_graph_and_geneset(G,common_features)
						X_ = X[:,[dataSet.genes_to_idx[n] for n in LCC_graph.nodes()] ]
						
						A = nx.adjacency_matrix(LCC_graph).todense()

						matrix_list = [A]
						matrix_names = ['raw']
						if weighting == 'weighted':
							A_normalized = A/np.amax(A)
							matrix_list.append(A_normalized)
							matrix_names.append('normalized')


						for matrixName, A in zip(matrix_names,matrix_list):						
							for maxScale in range(1,MAX_SCALES):
								for maxMoment in range(1,MAX_MOMENTS):
									for central in [True,False]:
										for operator in ['diffu']:

											transformer = DiffusionWMT(maxScale,maxMoment,A,operator,central)

											graphTransform = FunctionTransformer(transformer.computeTransform)
											
											base_pipeline = [ ('scale1',StandardScaler()),('graphTransform',graphTransform),('scale2',StandardScaler())]
											model_tuples = {'SVC':('clf',LinearSVC(C=1,dual='auto',class_weight = 'balanced')),
											 'LR':('clf', LogisticRegression(C=1,class_weight = 'balanced'))}


											for model_name in model_tuples.keys():
												classifier = [model_tuples[model_name]]
										
										
												model = Pipeline(base_pipeline+classifier)
												for i, (train_index, test_index) in enumerate(splitter.split(X_,y)):
													X_train, y_train = X_[train_index,:], y[train_index]
													X_test, y_test = X_[test_index,:], y[test_index]
										
									
													model.fit(X_train,y_train)

													bin_preds = model.predict(X_test)

													acc = accuracy_score(y_test, bin_preds)
													bal_acc = balanced_accuracy_score(y_test,bin_preds)
											
													if model_name == 'LR':
														prob_preds = model.predict_proba(X_test)
													try:
														roc_auc = roc_auc_score(y_test,prob_preds[:,1])
													except: 
														roc_auc = -1 
													else:
														roc_auc = -1

												results['iter'].append(i)
												results['drug'].append(drug)
												results['tissue'].append(tissue)
												results['max_moment'].append(maxMoment)
												results['sparsity'].append(sparsity)
												results['weighting'].append(weighting)
												results['geneset'].append(geneset)
												results['model'].append(model_name)
												results['max_scale'].append(maxScale)
												results['edge_weights'].append(matrixName)
												results['acc'].append(acc)
												results['bal_acc'].append(bal_acc)
												results['roc_auc'].append(roc_auc)
		df = pd.DataFrame(results)
		path = "../results/sensitivity/{d}".format(d=drug)
		os.makedirs(path,exist_ok = True)
		df.to_csv("{p}/{gex}_sensitivity.csv".format(p=path,gex = gex_units))
									


if __name__ == '__main__':
	main()