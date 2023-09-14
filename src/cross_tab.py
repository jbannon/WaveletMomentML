import matplotlib.pyplot as plt 
import seaborn as sns 
import io_utils, utils
import sys
import numpy as np
from collections import defaultdict 
import scipy.stats as sps
import pandas as pd 

gex_units = 'log_tpm'
base_path = "../data/expression/cri/"
results = defaultdict(list)


for drug in ['Atezo', 'Pembro', 'Ipi', 'Nivo']:
	target_gene = utils.TARGET_GENE_MAP[utils.DRUG_TARGET_MAP[drug]]
	results = pd.DataFrame()
	for tissue in utils.DRUG_TISSUE_MAP[drug]:


		dataSet = io_utils.load_ICIDataSet(base_path,drug, tissue, gex_units)
		gene_idx = dataSet.genes_to_idx[target_gene]
		X_ = dataSet.X[:,gene_idx]
		y = dataSet.y

		X_r = X_[np.where(y==1)[0]]
		X_nr = X_[np.where(y==0)[0]]

		# pval = sps.wilcoxon(X_r, X_nr)
		t = [tissue]*X_.shape[0]
		
		df = pd.DataFrame({'tissue':t,'target_expression':X_,'response':y})

		results = pd.concat([results,df],axis=0)
	sns.boxplot(x="tissue", y= "target_expression",
            hue= 'response',
            data = results).set(title = "Target ({t}) Expression for {d}".format(d = drug, t=utils.DRUG_TARGET_MAP[drug]),
            ylabel = 'Target Expression (Log2(TPM+1))', xlabel = 'Tissue')
	plt.savefig("../figs/{d}_comparison.png".format(d=drug))
	plt.close()