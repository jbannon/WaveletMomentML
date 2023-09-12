import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import os 
import sys 

L = os.listdir("../results/")


# for file in L:
# 	drug, model, tissue = file.split("_")
# 	tissue, _ = tissue.split(".")
# 	print(drug)
# 	print(model)
# 	print(tissue)
# 	fname = "../results/" + file
# 	df = pd.read_csv(fname)
# 	fig = sns.boxplot(x="dimRed", y="test_acc",
#             hue="dimRed",
#             data=df)
# 	fig.set(title = "Test Accuracy Predicting Response to {d} with {m} in {t}".format(d=drug,m=model, t=tissue),
# 		xlabel='Dimensionality Reduction Method', ylabel='Test Accuracy')
# 	plt.legend(title='Method')
# 	plt.savefig("../figs/{d}_{m}_{t}.png".format(d=drug,m=model, t=tissue))
# 	plt.close()


for file in L:
	drug, tissue, model = file.split("_")
	model,_ = model.split(".")
	tissue_string = "All Tissues" if tissue =='ALL' else tissue
	if model == 'dummies':
		fname = "../results/"+file
		df = pd.read_csv(fname)
		for split_type in ['stratif','kfold']:
			df_ = df[df['Split_Type']== split_type]
			split_string = "Stratified" if split_type == 'stratif' else "random_splits"

			fig = sns.boxplot(x="clf_type", y="test_acc", hue="clf_type", data=df_)
			fig.set(title = "Dummy Classifier Performance\n {d} in {t}. {s} Splits".format(d=drug, t=tissue,s=split_string),
				xlabel='Dummy Classifier Type', ylabel='Test Accuracy')
			plt.savefig("../figs/DUMMIES_{d}_{t}_{s}.png".format(d=drug, t=tissue_string,s=split_string))
			plt.close()
		