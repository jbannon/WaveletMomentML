from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import LeaveOneOut
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

from WMTransform import WaveletMomentTransform, DiffusionWMT
from collections import defaultdict
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, precision_recall_curve, accuracy_score

import tqdm 
from sklearn.dummy import DummyClassifier




def main(
	config:Dict
	)->None:

	
	drug, rng_seed, data_dir, rank_count_cutoff, fdr_string, n_iters, do_one_hop =\
	 	utils.unpack_parameters(config['EXPERIMENT_PARAMS'])
	
	edge_types, connectivity = utils.unpack_parameters(config['NETWORK_PARAMS'])
	
	J, p, central = utils.unpack_parameters(config['WAVELET_PARAMS'])
	
	rng = np.random.RandomState(seed = rng_seed)
	tissues = utils.DRUG_TISSUE_MAP[drug]
	
	model, preproc, cv_criterion = utils.unpack_parameters(config['MODEL_PARAMS'])


	
	


	

	for et in edge_types:
		for sp in connectivity:
			

			netfile = "../data/networks/{sp}/{et}.pickle".format(sp = sp, et = et)
			with open(netfile,"rb") as istream:
				PPI_Graph = pickle.load(istream)
			
			
			res_base = "../results/mono_classification/loo/{d}/{w}/{s}/".format(d=drug,w=et,s=sp)
			os.makedirs(res_base, exist_ok=True)
			

			for tissue in tissues:
				

				DE_file = "../data/genesets/{drug}_{tissue}_DE.csv".format(drug = drug, tissue = tissue)
				expr_file = utils.make_data_file_name(data_dir,drug,tissue,'expression')
				response_file = utils.make_data_file_name(data_dir,drug,tissue,'response')
				immune_feature_file = utils.make_data_file_name(data_dir, drug,tissue,'immune_features')
				


				expression_data = pd.read_csv(expr_file)
				response_data = pd.read_csv(response_file)
				immune_features = pd.read_csv(immune_feature_file)
				DE_genes = pd.read_csv(DE_file,index_col=0)
				immune_features = immune_features[['Run_ID','IMPRES','Miracle']]
				immune_features.columns =['Run_ID','IMPRES','MIRACLE']

				

				
				DE_genes = DE_genes[DE_genes["thresh"] == fdr_string]
				
				DE_genes = DE_genes[DE_genes['Rankcount']>=rank_count_cutoff]
				
				DE_genes['AvgRank'] = DE_genes['Ranksum']/DE_genes['Rankcount']
				
				DE_genes.sort_values('AvgRank',inplace=True)
				DE_genes.reset_index(drop=True,inplace=True)
				

				gene_list = list(DE_genes['Gene'].values)

				common_genes = list(set(gene_list).intersection(set(expression_data.columns[1:])))
				if do_one_hop:
					seeds = [x for x in common_genes if x in PPI_Graph.nodes()]
					one_hop = [x for x in seeds]
					for n in seeds:
						nbrs = PPI_Graph.neighbors(n)
						one_hop.extend([x for x in nbrs])
					common_genes = one_hop
				
				LCC_Graph = utils.harmonize_graph_and_geneset(PPI_Graph,common_genes)
				# print("LCC Nodes : {L}".format(L =len(LCC_Graph.nodes)))


				results = defaultdict(list)				
				
				for feature in tqdm.tqdm(['TARGET','IMPRES','MIRACLE','WM','WM_Norm']): #'DE','LCC'
					if feature in ['IMPRES','MIRACLE']:
						X = immune_features[feature].values.reshape(-1,1)
					elif feature == 'TARGET':
						target_gene = utils.TARGET_GENE_MAP[utils.DRUG_TARGET_MAP[drug]]
						X = np.log2(expression_data[target_gene].values + 1).reshape(-1,1)
					elif feature == 'DE':
						X = np.log2(expression_data[gene_list].values+1)
					elif feature == 'LCC':
						X = np.log2(expression_data[[x for x in list(LCC_Graph.nodes)]].values+1)
					elif feature in ['WM','WM_Norm']:
						X = np.log2(expression_data[[x for x in list(LCC_Graph.nodes)]].values+1)
						if feature == 'WM_Norm':
							norm_transform = Normalizer()
							X = norm_transform.fit_transform(X)
						A = nx.adjacency_matrix(LCC_Graph).todense()
						if et =='weighted':
							A = A/np.amax(A)
						DWMT = DiffusionWMT(numScales = J,maxMoment =p, adjacency_matrix = A, central=central)
						X = DWMT.computeTransform(X)

					y = response_data['Response'].values
					
					
					
					classifier, param_grid = utils.make_model_and_param_grid(model,preproc,weight_LR = False)							
					
					loo = LeaveOneOut()
					
					
					for i, (train_index, test_index) in tqdm.tqdm(enumerate(loo.split(X)),total = X.shape[0],leave=False):
						
						X_train, X_test = X[train_index,:], X[test_index,:]
						y_train, y_test = y[train_index], y[test_index]
						
						clf = GridSearchCV(classifier,param_grid,scoring = cv_criterion)
						clf.fit(X_train,y_train)
				
						
						

						pred_bins = clf.best_estimator_.predict(X_test)
						pred_probs = clf.best_estimator_.predict_proba(X_test)
						results['feature'].append(feature)
						results['i'].append(i)
						results['y_true'].append(y_test[0])
						results['pred_bin'].append(pred_bins[0])
						results['pred_prob'].append(pred_probs[:,1][0])


				df = pd.DataFrame(results)
				one_hop_string = "_OH" if do_one_hop else ""
				df.to_csv("{r}/loo{o}.csv".format(r=res_base,o=one_hop_string))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("-config",help="The config file for these experiments")
	args = parser.parse_args()



	with open(args.config) as file:
		config = yaml.safe_load(file)

	main(config)
	