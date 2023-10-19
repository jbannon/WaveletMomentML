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
	

	tissues = utils.DRUG_TISSUE_MAP[drug]
	

	for et in edge_types:
		for sp in connectivity:
			

			netfile = "../data/networks/{sp}/{et}.pickle".format(sp = sp, et = et)
			with open(netfile,"rb") as istream:
				PPI_Graph = pickle.load(istream)
			
			results = defaultdict(list)
			res_base = "../results/mono_classification/monte_carlo/{d}/{w}/{s}/".format(d=drug,w=et,s=sp)
			os.makedirs(res_base, exist_ok=True)


		
			for tissue in tissues:
				print(tissue)

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
				
				DE_genes.sort_values('Rankcount',inplace=True)
				

				DE_genes = DE_genes[DE_genes['Rankcount']>=rank_count_cutoff]
				
				# DE_genes['AvgRank'] = DE_genes['Ranksum']/DE_genes['Rankcount']
				
				# DE_genes.sort_values('AvgRank',inplace=True)
				# DE_genes.reset_index(drop=True,inplace=True)
				
				gene_list = list(DE_genes['Gene'].values)

				common_genes = list(set(gene_list).intersection(set(expression_data.columns[1:])))
				
				# issues = list(set(expression_data.columns[1:]).difference(set(gene_list)))
				print("Gene list length: {L}".format(L =len(gene_list)))
				print("Common Genes length: {L}".format(L =len(common_genes)))

				# print(len(PPI_Graph.nodes))
				# cn = list(set(PPI_Graph.nodes).intersection(set(common_genes)))
				# print("Common PPI nodes: {L}".format(L =len(cn)))
				# diff = [x for x in common_genes if x not in cn]
				if do_one_hop:
					seeds = [x for x in common_genes if x in PPI_Graph.nodes()]
					one_hop = [x for x in seeds]
					for n in seeds:
						nbrs = PPI_Graph.neighbors(n)
						one_hop.extend([x for x in nbrs])
					common_genes = one_hop
				
					
				LCC_Graph = utils.harmonize_graph_and_geneset(PPI_Graph,common_genes)
				print("LCC Nodes : {L}".format(L =len(LCC_Graph.nodes)))
				


				



				for feature in tqdm.tqdm(['TARGET','IMPRES','MIRACLE','DE','LCC','WM','WM_Norm']):
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
				
					for balance_weights in [True,False]:
						
						classifier, param_grid = utils.make_model_and_param_grid(model,preproc,weight_LR = balance_weights)							

						for i in tqdm.tqdm(range(n_iters),leave = False):
							X_train, X_test, y_train, y_test =\
								train_test_split(X,y,test_size = 0.2, stratify=y)
							
							try:
								clf = GridSearchCV(classifier,param_grid,scoring = cv_criterion, verbose = 0)
								clf.fit(X_train,y_train)
							except:
								print(y_train)
								print("grid")
								sys.exit(1)
							
							

							pred_bins = clf.best_estimator_.predict(X_test)
							pred_probs = clf.best_estimator_.predict_proba(X_test)
							

							# binary results
							acc = accuracy_score(y_test,pred_bins)
							tn, fp, fn, tp = confusion_matrix(y_test, pred_bins,labels = [0,1]).ravel()
							
							try:
								roc_auc = roc_auc_score(y_test,pred_probs[:,1])
							except:
								print(y_train)
								sys.exit(1)
							brier = brier_score_loss(y_test, pred_probs[:,1])
							logloss = log_loss(y_test,pred_probs[:,1])
							
							
							results['iter'].append(i)
							results['drug'].append(drug)
							results['tissue'].append(tissue)
							results['model'].append(model)
							results['penalty'].append(clf.best_params_['clf__C'])
							results['preproc'].append(preproc)
							
							results['balance_weights'].append(balance_weights)
							results['feature'].append(feature)
							results['feat_dim'].append(X_train.shape[1])



							results['acc'].append(acc)
							results['tp'].append(tp)
							results['tn'].append(tn)
							results['fp'].append(fp)
							results['fn'].append(fn)		
							results['roc_auc'].append(roc_auc)
							results['brier'].append(brier)
							results['log_loss'].append(logloss)

							
							
				df = pd.DataFrame(results)
				one_hop_string = "_OH" if do_one_hop else ""
				df.to_csv("{r}/mono_classification{o}.csv".format(r=res_base,o=one_hop_string))
			
	



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("-config",help="The config file for these experiments")
	args = parser.parse_args()



	with open(args.config) as file:
		config = yaml.safe_load(file)

	main(config)
	