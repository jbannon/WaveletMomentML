from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import brier_score_loss, log_loss
import sys 
import os 
import argparse
import yaml 
from typing import Union, List, Dict, Tuple
# import io_utils, utils, model_utils, geneset_utils, graph_utils
import numpy as np
import pickle 
import networkx as nx 
import pandas as pd
import utils


from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from WMTransform import WaveletMomentTransform, DiffusionWMT
from collections import defaultdict
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, precision_recall_curve, accuracy_score,balanced_accuracy_score

import tqdm 
from sklearn.dummy import DummyClassifier

from sklearn.preprocessing import Normalizer


def main():
	data_dir = "../data/expression/"
	MAX_SCALES = 3
	MAX_MOMENTS = 4
	n_iter = 10
	preproc_map= {'center':('preproc',StandardScaler(with_std=False)),'standardize':('preproc',StandardScaler()),
		'unit_norm':('preproc',Normalizer())}

	postproc_map= {'robust':('postproc',RobustScaler()),
		'center':('postproc',StandardScaler(with_std=False)),'standardize':('postproc',StandardScaler())}
	



	
	for drug in utils.DRUG_TISSUE_MAP.keys():
		results = defaultdict(list)
		


		for sparsity in ['sparse','dense']:
			for weighting in ['weighted','unweighted']:
			
				netfile = "../data/networks/{sp}/{w}.pickle".format(sp=sparsity, w=weighting)
				
				with open(netfile,"rb") as istream:
					PPI_Graph = pickle.load(istream)
				
				for tissue in utils.DRUG_TISSUE_MAP[drug]:
					DE_file = "../data/genesets/{drug}_{tissue}_DE.csv".format(drug = drug, tissue = tissue)
					expr_file = utils.make_data_file_name(data_dir,drug,tissue,'expression')
					response_file = utils.make_data_file_name(data_dir,drug,tissue,'response')
					expression_data = pd.read_csv(expr_file)
					response_data = pd.read_csv(response_file)
					
					print(response_data['Run_ID'].eq(expression_data['Run_ID']).all())
					
					DE_genes = pd.read_csv(DE_file)
					# DE_genes = DE_genes[DE_genes['Rankcount']>=25]
					DE_genes = DE_genes[DE_genes["thresh"]== 'strict']
					# DE_genes['WeightedAvgRank'] = DE_genes['Ranksum']*(DE_genes['total.iter']/DE_genes['Rankcount'])
					
					# DE_genes.sort_values('WeightedAvgRank',inplace=True)
						
					
					gene_list = list(DE_genes['Gene'].values)
					
					common_genes = list(set(gene_list).intersection(set(expression_data.columns[1:])))
					
					
					LCC_Graph = utils.harmonize_graph_and_geneset(PPI_Graph,common_genes)

					X = np.log2(expression_data[[x for x in LCC_Graph.nodes()]].values+1)
					

					y = response_data['Response'].values
					
					A = nx.adjacency_matrix(LCC_Graph).todense()

					matrix_list = [A]
					matrix_names = ['raw']
					
					if weighting == 'weighted':
						A_normalized = A/np.amax(A)
						matrix_list.append(A_normalized)
						A_string = A/1000
						matrix_list.append(A_string)
					
					matrix_names.append('normalized')
					matrix_names.append('string')


					for matrixName, A in zip(matrix_names,matrix_list):						
						for maxScale in range(1,MAX_SCALES):
							for maxMoment in range(1,MAX_MOMENTS):
								for central in [True,False]:
									for operator in ['d','g']:
										transformer = DiffusionWMT(maxScale,maxMoment,A,operator,central)
										
										for preproc in preproc_map.keys():
											for postproc in postproc_map.keys():
												print("preproc: {p}".format(p=preproc))
												print("postproc: {p}".format(p=postproc))
												print("operator: {o}".format(o=operator))
												print("maxScale {j}".format(j=maxScale))
												print("maxMoment {j}".format(j=maxMoment))
												print("central moments? {c}".format(c=central))
												print("matrix: {m}".format(m=matrixName))
												pre_step = preproc_map[preproc]
												post_step = postproc_map[postproc]
												wave_step = ("wavemoment",FunctionTransformer(transformer.computeTransform))
												model_step = ('clf',LogisticRegression(C = 0.5))

												model = Pipeline([pre_step,wave_step,post_step,model_step])
												
												for i in tqdm.tqdm(range(n_iter),leave=False):
													X_train, X_test, y_train, y_test =\
														train_test_split(X,y,test_size = 0.8, stratify=y)

													model.fit(X_train,y_train)
													pred_bins = model.predict(X_test)
													pred_probs = model.predict_proba(X_test)
													

													acc = accuracy_score(y_test,pred_bins)
													tn, fp, fn, tp = confusion_matrix(y_test, pred_bins,labels = [0,1]).ravel()
													
													roc_auc = roc_auc_score(y_test,pred_probs[:,1])
													brier = brier_score_loss(y_test, pred_probs[:,1])
													logloss = log_loss(y_test,pred_probs[:,1])

													bal_acc = balanced_accuracy_score(y_test,pred_bins)
									
												
													results['iter'].append(i)
													results['drug'].append(drug)
													results['tissue'].append(tissue)
													results['preproc'].append(preproc)
													
													results['postproc'].append(postproc)
													results['max_scale'].append(maxScale)
													results['max_moment'].append(maxMoment)
													results['weighting'].append(weighting)
													results['sparsity'].append(sparsity)
													results['matrix'].append(matrixName)
													


													results['acc'].append(acc)
													results['bal_acc'].append(bal_acc)
													results['tp'].append(tp)
													results['tn'].append(tn)
													results['fp'].append(fp)
													results['fn'].append(fn)		
													results['roc_auc'].append(roc_auc)
													results['brier'].append(brier)
													results['log_loss'].append(logloss)

						df = pd.DataFrame(results)
						df.to_csv("bingbong.csv")
						sys.exit(1)
		path = "../results/sensitivity/{d}".format(d=drug)
		os.makedirs(path,exist_ok = True)
		df.to_csv("{p}/{gex}_sensitivity.csv".format(p=path,gex = gex_units))
									


if __name__ == '__main__':
	main()