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




def main(
	config:Dict
	)->None:
	

	"""
	Experimental Parameters
	"""
	
	drug: str
	rngSeed: int 
	dataDir: str
	countCutoff:int 
	pvalThresh: float 
	numIters: int 
	tissues: List[str]
	doOneHop: bool 
	testSize: float

	"""
	Network Parameters 
	"""
	edgeTypes: List[str]
	graphDensities:List[str]
	networkNorm:str
	

	"""
	Wavelet Parameters
	"""
	numScales:int 
	maxMoment:int
	central: str
	waveletType: str

	""" 
	Model Parameters
	"""
	model: str
	preprocessingSteps: List[str]
	cvCriterion:str
	

	drug, rngSeed, dataDir, countCutoff, pvalThresh, numIters, tissues, doOneHop =\
	 	utils.unpack_parameters(config['EXPERIMENT_PARAMS'])

	edgeTypes, graphDensities, networkNorm = utils.unpack_parameters(config['NETWORK_PARAMS'])
	
	numScales, maxMoment, central, waveletType = utils.unpack_parameters(config['WAVELET_PARAMS'])	
	
	model, preprocessingSteps, cvCriterion = utils.unpack_parameters(config['MODEL_PARAMS'])
	
	rng = np.random.RandomState(seed = rngSeed)
	
	for edgeType in edgeTypes:
		for density in graphDensities:
			
			netfile = "../data/networks/{sp}/{et}.pickle".format(sp = density, et = edgeType)
			
			with open(netfile,"rb") as istream:
				PPI_Graph = pickle.load(istream)
			
			results = defaultdict(list)
			res_base = "../results/mono_classification/monte_carlo/{d}/{e}/{c}/{w}/".format(d=drug,e=edgeType,c=density,w=waveletType)
			os.makedirs(res_base, exist_ok=True)


		
			for tissue in tqdm.tqdm(tissues,total = len(tissues)):
				# print(tissue)

				DE_file = "../data/genesets/{drug}_{tissue}_DE.csv".format(drug = drug, tissue = tissue)
				expr_file = utils.make_data_file_name(dataDir,drug,tissue,'expression')
				response_file = utils.make_data_file_name(dataDir,drug,tissue,'response')
				immune_feature_file = utils.make_data_file_name(dataDir, drug,tissue,'immune_features')
				


				expression_data = pd.read_csv(expr_file)
				response_data = pd.read_csv(response_file)
				immune_features = pd.read_csv(immune_feature_file)
				DE_genes = pd.read_csv(DE_file,index_col=0)
				immune_features = immune_features[['Run_ID','IMPRES','Miracle']]
				immune_features.columns =['Run_ID','IMPRES','MIRACLE']

				
				
				
				DE_genes = DE_genes[DE_genes["Thresh.Value"] == pvalThresh]
				DE_genes = DE_genes[DE_genes['Count']>=countCutoff]
				gene_list = list(DE_genes['Gene'].values)

				# common_genes = list(set(gene_list).intersection(set(expression_data.columns[1:])))
				common_genes = [x for x in gene_list if x in list(expression_data.columns[1:])]
				# issues = list(set(expression_data.columns[1:]).difference(set(gene_list)))
				# print("Gene list length: {L}".format(L =len(gene_list)))
				# print("Common Genes length: {L}".format(L =len(common_genes)))

	
				if doOneHop:
					seeds = [x for x in common_genes if x in PPI_Graph.nodes()]
					one_hop = [x for x in seeds]
					for n in seeds:
						nbrs = PPI_Graph.neighbors(n)
						one_hop.extend([x for x in nbrs])
					common_genes = one_hop
				
					
				LCC_Graph = utils.harmonize_graph_and_geneset(PPI_Graph,common_genes)
				# print("LCC Nodes : {L}".format(L =len(LCC_Graph.nodes)))
				


				


				
				for feature in tqdm.tqdm(['WM','WM_Norm','TARGET','IMPRES',
					'MIRACLE','LCC','DE','WM_Robust','WM_Standard', 'WM_Center'], leave = False):
					if feature in ['IMPRES','MIRACLE']:
						X = immune_features[feature].values.reshape(-1,1)
					elif feature == 'TARGET':
						target_gene = utils.TARGET_GENE_MAP[utils.DRUG_TARGET_MAP[drug]]
						X = np.log2(expression_data[target_gene].values + 1).reshape(-1,1)
					elif feature == 'DE':
						X = np.log2(expression_data[gene_list].values+1)
					elif feature == 'LCC':
						X = np.log2(expression_data[[x for x in list(LCC_Graph.nodes)]].values+1)
					elif feature in ['WM','WM_Norm','WM_Robust','WM_Standard']:
						X = np.log2(expression_data[[x for x in list(LCC_Graph.nodes)]].values+1)
						if feature == 'WM_Norm':
							transform = Normalizer()
							X = transform.fit_transform(X)
						elif feature == 'WM_Robust':
							transform = RobustScaler()
							X = transform.fit_transform(X)
						elif feature == 'WM_Standard':
							transform = StandardScaler()
							X = transform.fit_transform(X)
						elif feature == 'WM_Center':
							transform = StandardScaler(with_std = False)
							X = transform.fit_transform(X)

						A = nx.adjacency_matrix(LCC_Graph).todense()
						if edgeType =='weighted':
							scale_factor = 1
							scale_factor = 1000 if networkNorm.upper() == "STRING" else scale_factor
							scale_factor = np.amax(A) if networkNorm.upper() == "MAX" else scale_factor
							A = A/scale_factor

						if waveletType.lower()=='diffusion':
							WMT = DiffusionWMT(numScales = numScales,
							maxMoment = maxMoment, 
							adjacency_matrix = A, 
							central=central)
						elif waveletType.lower() == 'geometric':
							WMT = GeometricWMT(numScales = numScales,
							maxMoment = maxMoment, 
							adjacency_matrix = A, 
							central=central)

						X = WMT.computeTransform(X)

					y = response_data['Response'].values
				
					for balanceWeights in [True,False]:
						for step in preprocessingSteps:
							classifier, paramGrid = utils.make_model_and_param_grid(model,step,weight_LR = balanceWeights)							
						
							for i in tqdm.tqdm(range(numIters),leave = False):
								X_train, X_test, y_train, y_test =\
									train_test_split(X,y,test_size = 0.25, stratify=y)
								
								try:
									clf = GridSearchCV(classifier,paramGrid,scoring = cvCriterion, verbose = 0)
									clf.fit(X_train,y_train)
								except Exception as error:
									print(error)
									print(step)
									print(feature)
									print(np.isnan(X))
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
									print(np.amax(X))
									sys.exit(1)
								
								brier = brier_score_loss(y_test, pred_probs[:,1])
								logloss = log_loss(y_test,pred_probs[:,1])
								
								
								results['iter'].append(i)
								results['drug'].append(drug)
								results['tissue'].append(tissue)
								results['model'].append(model)
								results['penalty'].append(clf.best_params_['clf__C'])
								results['preproc'].append(step)
								results['J'].append(numScales)
								results['p'].append(maxMoment)
								results['balance_weights'].append(balanceWeights)
								results['feature'].append(feature)
								results['feat_dim'].append(X_train.shape[1])
								results['network_weight'].append(edgeType)
								results['connectivity'].append(density)

								results['acc'].append(acc)
								results['tp'].append(tp)
								results['tn'].append(tn)
								results['fp'].append(fp)
								results['fn'].append(fn)		
								results['roc_auc'].append(roc_auc)
								results['brier'].append(brier)
								results['log_loss'].append(logloss)

							
							
				df = pd.DataFrame(results)
				one_hop_string = "_OH" if doOneHop else ""
				df.to_csv("{r}/monte_carlo{o}.csv".format(w= waveletType, r=res_base,o=one_hop_string))
			
	



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("-config",help="The config file for these experiments")
	args = parser.parse_args()



	with open(args.config) as file:
		config = yaml.safe_load(file)

	main(config)
	