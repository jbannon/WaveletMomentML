from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import networkx as nx
from typing import List, Tuple, Dict, NamedTuple
from collections import namedtuple, defaultdict
import numpy as np
import sys
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
    

    common_genes = list(set(G.nodes).intersection(set(gene_set)))
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
    preproc:str = 'standardize',
    weight_LR:bool = True,
    pca_dims:int = 5,
    doPCA:bool = False):
    
    
    assert preproc.lower() in ['center','robust','standardize']
    
    if model_name == 'KNN':
        model = ('clf', KNeighborsClassifier())
        param_grid = {'n_neighbors':np.arange(2,11,2)}

    elif model_name == 'LR':
        model = ('clf', LogisticRegression())
        if weight_LR:
            cw = ['balanced']
        else:
            cw = [None]
        param_grid = {'penalty':['l2'], 'max_iter':[10**10], 'solver':['lbfgs'],
             'C':np.arange(1,101)*(10**-2),'class_weight':cw,'verbose':[0]}
        
    
    
    if preproc.lower()=='center':
        preproc = [('scaler',StandardScaler(with_std=False))]
    elif preproc.lower()=='robust':
        preproc = [('scaler',RobustScaler())]
    elif preproc.lower()=='standardize':
        preproc = [('scaler',StandardScaler())]

    model = preproc+[model]
    param_grid = {"clf__"+k:param_grid[k] for k in param_grid}

    return Pipeline(model), param_grid



