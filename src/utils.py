from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import networkx as nx
from typing import List, Tuple, Dict, NamedTuple
from collections import namedtuple, defaultdict
import numpy as np
import sys
from scipy.stats import kurtosis,mode, skew, beta
from typing import List, Tuple, Dict, NamedTuple
import numpy as np 
import pickle as pk
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA

DRUG_TISSUE_MAP = {'Atezo':['KIRC','BLCA'],
                   'Ipi':['SKCM'],
                   'Nivo':['SKCM'],
                   'Pembro':['SKCM','STAD']
                    }


DRUG_TARGET_MAP = {'Atezo':'PD-L1','Pembro':'PD1','Nivo':'PD1','Ipi':'CTLA4'}


TARGET_GENE_MAP = {'PD-L1':'CD274', 'PD1':'PDCD1', 'CTLA4':'CTLA4'}



NETBIO_MAP = {'CD8T':'T_exhaust_Pos','TAM':'TAM_M2_M1_Pos','TME':'all-TME-Bio'}

def make_data_file_name(base_dir:str, drug:str, tissue:str, data_type:str):
    

    base_dir = base_dir[:-1] if base_dir[-1]=="/" else base_dir
    fname = "{b}/{d}/{t}/{dt}.csv".format(b=base_dir,d=drug,t=tissue,dt=data_type)
    return fname

def unpack_parameters(
    D:Dict
    ):
    if len(D.values())>1:
        return tuple(D.values())
    else:
        return tuple(D.values())[0]


def harmonize_graph_and_geneset(
    G:nx.Graph,
    gene_set:List[str]
    ) -> nx.Graph:
    


    common_genes = [x for x in list(G.nodes) if x in gene_set]
    # print(len(common_genes))


    # for cc in nx.connected_components(G):
    #     for g in common_genes:
    #         if g in cc:
    #             print("{g}\t{cc}".format(g=g,cc=len(cc)))
    
    H = G.subgraph(common_genes)
    # print(len(H.nodes))
    
    # for cc in nx.connected_components(H):
    #     if len(cc)==1:
    #         print(cc)
    
    if not nx.is_connected(H):
        LCC_genes = max(nx.connected_components(H), key=len)
        H = H.subgraph(LCC_genes)
    
    return H


def make_model_and_param_grid(
    model_name:str,
    do_pca:bool = False,
    pca_dim:int = 3,
    preproc:str = 'standardize'):
    
    
    
    
    if model_name == 'KNN':
        model = ('clf', KNeighborsClassifier())
        param_grid = {'n_neighbors':np.arange(2,11,2)}

    elif model_name == 'LR':
        model = ('clf', LogisticRegression())
        param_grid = {'penalty':['l2'], 'max_iter':[10**10], 'solver':['lbfgs'],
             'C':np.arange(1,101)*(10**-2),'class_weight':['balanced'],'verbose':[0]}
        
    
    
    if preproc.lower()=='center':
        preproc = [('scaler',StandardScaler(with_std=False))]
    elif preproc.lower()=='robust':
        preproc = [('scaler',RobustScaler())]
    elif preproc.lower()=='standardize':
        preproc = [('scaler',StandardScaler())]

    if do_pca:
        preproc = preproc + [('dim_red',PCA(n_components = pca_dim))]
    model = preproc+[model]
    param_grid = {"clf__"+k:param_grid[k] for k in param_grid}

    return Pipeline(model), param_grid



def fetch_drug_targets(
    drug:str
    ) -> List[str]:
    if drug in DRUG_TARGET_MAP.keys():
        targets = [TARGET_GENE_MAP[DRUG_TARGET_MAP[drug]]]
    else:
        fname = "../data/genesets/{d}_targets.txt".format(d=drug)
        with open(fname, "r") as istream:
            lines = istream.readlines()
        targets = [x.rstrip() for x in lines]

    return targets


def empirical_bayes_gene_selection(
    df:pd.DataFrame,
    prob_thresh:float,
    n_runs:int = 200):
    

    temp = df.copy(deep=True)
    
    temp["Hits"] = temp['Count']
    temp['Misses'] = n_runs-temp['Count']
    temp['Proportion'] = temp["Count"]/n_runs

    a, b,loc, scale = beta.fit(temp[temp["Proportion"]<1.0]["Proportion"].values,fscale = 1, floc = 0)

    temp["EB_avg"] = (temp["Hits"]+a)/(n_runs+a+b)
            
    EB_point = np.amin(temp[temp["EB_avg"]>=prob_thresh]["Count"].values)
            
    temp = temp[temp['Count'] >= EB_point]
    # prob_estimates = defaultdict(list)
    # # map CDFs
    # for idx, row in temp.iterrows():            
    #     a1 = a + row["Hits"]
    #     b1 = b + row["Misses"]
    #     this_Beta = beta(a=a1,b=b1)
    #     this_Prob = this_Beta.sf(prob_thresh)
    #     prob_estimates['Gene'].append(row['Gene'])
    #     prob_estimates['EB_Prob'].append(thisProb)
    # pb = pd.DataFrame(prob_estimates)
    # EB = pb[pb['EB_Prob']>=conf]
    # EB = temp[temp["Gene"].isin(EB["Gene"].values)]

    # if cutType == 'EB':
    #     temp = temp[temp['Count'] >= EB_point]
    # elif cutType == "EBCDF":
    #     try:
    #         cut = np.amin(EB["Count"].values)
    #         temp = temp[temp['Count'] >= cut]
    #     except: 
    #         print("cdf causing an issue, using prob estimate")
            

    gene_list = list(temp['Gene'].values)
    return gene_list

    

