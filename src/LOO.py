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
	do_one_hop: bool 
	test_size: float
	num_genes:int

	"""
	Network Parameters 
	"""
	edge_types: List[str]
	graph_densities:List[str]
	edge_norm_method:str
	

	"""
	Wavelet Parameters
	"""
	num_scales:int 
	max_moment:int
	central: str
	wavelet_type: str

	""" 
	Model Parameters
	"""
	model: str
	preprocessing_steps: List[str]
	cv_criterion:str
	

	drug, rng_seed, data_dir, pval_thresh, prob_thresh, train_pct, num_iters, tissues, do_one_hop, num_genes =\
	 	utils.unpack_parameters(config['EXPERIMENT_PARAMS'])

	net_dir, edge_types, graph_densities, network_norm = utils.unpack_parameters(config['NETWORK_PARAMS'])
	
	num_scales, max_moment, central, wavelet_type = utils.unpack_parameters(config['WAVELET_PARAMS'])	
	
	models,  cv_criterion = utils.unpack_parameters(config['MODEL_PARAMS'])
	
	rng = np.random.RandomState(seed = rng_seed)
	

	for tissue in tqdm.tqdm(tissues,total = len(tissues)):

		DE_file = "../data/genesets/{drug}_{tissue}_DE.csv".format(drug = drug, tissue = tissue)
		expr_file = utils.make_data_file_name(data_dir,drug,tissue,'expression')
		response_file = utils.make_data_file_name(data_dir,drug,tissue,'response')
		
		
		expression_data = pd.read_csv(expr_file)
		response_data = pd.read_csv(response_file)	
		DE_data = pd.read_csv(DE_file,index_col=0)
		
		
		
		
		
		DE_data = DE_data[DE_data["Thresh.Value"] == pval_thresh]

		
		gene_list = utils.empirical_bayes_gene_selection(DE_data,prob_thresh)

		common_genes = [x for x in gene_list if x in list(expression_data.columns[1:])]
		results = defaultdict(list)
		res_dir = "../results/loo/{d}/{t}".format(d=drug,t=tissue)
		os.makedirs(res_dir, exist_ok=True)
		
		for edge_type in edge_types:
			for density in graph_densities:
			
				netfile = "{n}/{sp}/{et}.pickle".format(n=net_dir,sp = density, et = edge_type)
			
				with open(netfile,"rb") as istream:
					PPI_Graph = pickle.load(istream)
			
				results = defaultdict(list)
				
				
	
				if do_one_hop:
					seeds = [x for x in common_genes if x in PPI_Graph.nodes()]
					one_hop = [x for x in seeds]
					for n in seeds:
						nbrs = PPI_Graph.neighbors(n)
						one_hop.extend([x for x in nbrs])
					working_geneset = one_hop
				else:
					working_geneset = [x for x in common_genes]
					
				LCC_Graph = utils.harmonize_graph_and_geneset(PPI_Graph,working_geneset)
				# print("LCC Nodes : {L}".format(L =len(LCC_Graph.nodes)))
				


				

				
				
				for feature in tqdm.tqdm(['WM','DE','LCC','LCC-PCA'],leave = False):

					if feature == 'DE':
						X = np.log2(expression_data[gene_list].values+1)
					elif feature in ['LCC','LCC-PCA']:
						X = np.log2(expression_data[[x for x in list(LCC_Graph.nodes)]].values+1)
					elif feature in ['WM']:
						X = np.log2(expression_data[[x for x in list(LCC_Graph.nodes)]].values+1)

						A = nx.adjacency_matrix(LCC_Graph).todense()
						if edge_type =='weighted':
							scale_factor = 1
							scale_factor = 100 if network_norm.upper() == "STRING" else scale_factor
							scale_factor = np.amax(A) if network_norm.upper() == "MAX" else scale_factor
							A = A/scale_factor

						
						WMT = DiffusionWMT(numScales = num_scales,
							maxMoment = max_moment, 
							adjacency_matrix = A, 
							central=central)
						X = WMT.computeTransform(X)

					y = response_data['Response'].values

				


					for model in models:
						if feature == 'LCC-PCA':
							pca_dim = max_moment*num_scales
							classifier, param_grid = utils.make_model_and_param_grid(model,do_pca =True, pca_dim = pca_dim)							
						else:
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
							results['edge_type'].append(edge_type)
							results['i'].append(i)
							results['tissue'].append(tissue)
							results['y_true'].append(y_test[0])
							results['pred_bin'].append(pred_bins[0])
							results['pred_prob'].append(pred_probs[:,1][0])

							
							
		df = pd.DataFrame(results)
		one_hop_string = "_OH" if do_one_hop else ""
		df.to_csv("{r}/wavelets_{o}.csv".format(r=res_dir,o=one_hop_string))
			
	



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("-config",help="The config file for these experiments")
	args = parser.parse_args()



	with open(args.config) as file:
		config = yaml.safe_load(file)

	main(config)
	