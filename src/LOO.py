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
	rngSeed: int 
	dataDir: str
	countCutoff:int 
	pvalThresh: float 
	numIters: int 
	tissues: List[str]
	doOneHop: bool 
	testSize: float
	nGenes:int

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
	

	drug, rngSeed, dataDir, countCutoff, pvalThresh, numIters, tissues, doOneHop, nGenes =\
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
			res_base = "../results/mono_classification/loo/{d}/{e}/{c}/{w}/".format(d=drug,e=edgeType,c=density,w=waveletType)
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

				q = (1.0*nGenes)/len(pd.unique(DE_genes['Gene']))
				thresh = 1-q
				qval = DE_genes['Count'].quantile(thresh)
					



				DE_genes = DE_genes[DE_genes['Count']>=qval]
				gene_list = list(DE_genes['Gene'].values)

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
				


				

				
				
				for feature in tqdm.tqdm(['WM','WM_Norm','TARGET','IMPRES','DE',
					'MIRACLE','LCC','WM_Robust','WM_Standard', 'WM_Center'],leave = False):

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
							loo = LeaveOneOut()
							
							for i, (train_index, test_index) in tqdm.tqdm(enumerate(loo.split(X)),
								total = X.shape[0],leave=False):
						
								X_train, X_test = X[train_index,:], X[test_index,:]
								y_train, y_test = y[train_index], y[test_index]
								
								clf = GridSearchCV(classifier,paramGrid,scoring = cvCriterion)
								clf.fit(X_train,y_train)
						
								
								

								pred_bins = clf.best_estimator_.predict(X_test)
								pred_probs = clf.best_estimator_.predict_proba(X_test)
								results['feature'].append(feature)
								results['i'].append(i)
								results['tissue'].append(tissue)
								results['y_true'].append(y_test[0])
								results['pred_bin'].append(pred_bins[0])
								results['pred_prob'].append(pred_probs[:,1][0])

							
							
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
	