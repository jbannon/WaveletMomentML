import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import os 
import sys 

L = os.listdir("../results/")


df = pd.DataFrame()
for t in ['ALL', 'SKCM','STAD']:
	temp = pd.read_csv("../results/mono_classification/montecarlo/log_tpm/Pembro/{t}/results.csv".format(t=t))
	df = pd.concat([df,temp],axis=0)




print(df.head())
print(df.columns)
print(pd.unique(df['model']))
print(pd.unique(df['feature']))
print(df.shape)



# change to [geneset, transform]

dummy_models = ['most_frequent', 'uniform' ,'stratified']

# for 

dummy_info = df[df['model'].isin(dummy_models)]

print(dummy_info.columns)


# for measure in ['acc','bal_acc','f1']:
# 	temp = dummy_info[['tissue','model',measure]]
# 	sns.boxplot(x="tissue", y= measure,
#             hue= 'model',
#             data=temp)
# 	plt.show()


auslander_models = ['auslander_common', 'auslander_LCC', 'auslander_wm']

df_ = df[df['feature'].isin(auslander_models)]
for measure in ['acc','bal_acc','f1']:
	for model in ['LR','RBF','SVC']:
		temp = df_[df_['model']==model]
		sns.boxplot(x="tissue", y= measure,
            hue= 'feature',
            data = temp).set(title = model)
		plt.savefig("../figs/{m1}_{m2}.png".format(m1 = model, m2 = measure))
		plt.close()


