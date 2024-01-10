import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import os 
import sys











# print(df)
# print(pd.unique(df['feature']))
for drug in ['Pembro']:
	for topology in ['dense','sparse','tight']:
		for waveletType in ['Diffusion','Geometric']:
			for oneHop in [True,False]:
				
				if oneHop:
					oh_string = "_OHome "
					ohDir = "OneHop"
				else:
					oh_string = ""
					ohDir = "NoHop"
				fname = "../results/mono_classification/loo/{drg}/weighted/{top}/{wav}/leave_one_out{oh}.csv".format(drg = drug,top=topology, wav= waveletType, oh=oh_string)
				df = pd.read_csv(fname)
				
				for tissue in pd.unique(df['tissue']):
			
					temp1 = df[df['tissue']==tissue]
					for feat in ['WM','LCC','TARGET']:
					
						temp = temp1[temp1['feature']==feat]
						fpr, tpr, thresholds = roc_curve(temp['y_true'].values, temp['pred_prob'].values, pos_label=1)
						auc = roc_auc_score(temp['y_true'], temp['pred_prob'])
						plt.plot(fpr,tpr,label = "{f} - {s}".format(f=feat,s=np.round(auc,2)))


					plt.plot([0, 1], ls="--")
					plt.legend(loc="upper left")
					oh = 'One Hop' if oneHop else 'No Hop'
					title = "{d} {t} {w} {oh}".format(d=drug,t=tissue, w= waveletType, oh=oh)
					fpath = "../figs/loo/{d}/{t}/{top}/{w}/{oh}/".format(d=drug,t=tissue,top=topology, w= waveletType, oh=ohDir)
					os.makedirs(fpath, exist_ok = True)

					figname = fpath + "results.png"
					print(figname)
					plt.title(title)
					plt.savefig(figname)
					# plt.show()
					plt.close()
					# sys.exit(1)
					

