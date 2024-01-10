import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import os 
import sys









df = pd.read_csv("../results/mono_classification/monte_carlo/Atezo/weighted/sparse/mono_classification_OH.csv")

drug = 'Atezo'

ms_map = {'acc':'Accuracy', 'roc_auc': "ROC_AUC", 'brier':'Brier','log_loss':'Log Loss'}

for tissue in pd.unique(df['tissue']):
	fig_path = "../figs/mono_classification/monte_carlo/{d}/{t}".format(d=drug,t=tissue)
	os.makedirs(fig_path,exist_ok=True)
	for balance in pd.unique(df['balance_weights']):
		bal_string = 'Balanced Weights' if balance else 'Uniform Weights'
		temp = df[(df['tissue']==tissue) & (df['balance_weights']==balance)]
		
		for measure in ['acc','roc_auc','brier','log_loss']:
			
			dims = temp.groupby(['feature'])['feat_dim'].mean()
			# print(dims)
			vertical_offset = temp[measure].median() * 0.05 	
			ascend = measure in ['acc','roc_auc']
			group_means=temp.groupby(['feature'])[measure].mean().sort_values(ascending = ascend)
			
			ax = sns.boxplot(x= 'feature', y = measure, data = temp, order=group_means.index)
			
			ax.set(title = "{d} {t} {m} {b}".format(d=drug, t=tissue, m = ms_map[measure],b = bal_string),
				xlabel = "Feature", ylabel = "Test {m}".format(m = ms_map[measure]))
			
			max_y = max([y for y in ax.get_yticks()])
			
			
			for d in dims.index:
				for xtick in ax.get_xticks():				
					name = group_means.index[xtick]
					n_dim = int(dims[name])
					ax.text(xtick,max_y - 0.055, n_dim, 
						horizontalalignment='center',size='x-small',color='black',weight='semibold')
			
			plt.savefig("{p}/{m}_{b}_OH.png".format(p=fig_path, m =ms_map[measure],b = bal_string))
			plt.close()
