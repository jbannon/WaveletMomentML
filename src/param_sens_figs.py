import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd






df = pd.read_csv("../results/dense_weighted_KIRC.csv")
df['tnr'] = 1.0*df['tn']/(df['tn']+df['fp'])
raw = df[df['matrixName']=='raw']

measure = 'tpr'
pp = raw[['scale','moment',measure]].groupby(['scale','moment'])[measure].mean()

pp = pp.reset_index().pivot(columns='scale',index='moment',values=measure)


print(pp)
sns.heatmap(pp)
plt.show()
# pp = raw[['scale','moment','bal_acc']].pivot(index="scale", columns="moment", values="bal_acc")

# print(pp)
# plt.plot()