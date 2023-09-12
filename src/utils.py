from typing import List, Tuple, Dict, NamedTuple
from collections import namedtuple, defaultdict
import numpy as np
import sys
from typing import List, Tuple, Dict, NamedTuple
import numpy as np 
import pickle as pk


   
class ICIDataSet(NamedTuple):
    drug:str 
    tissue:str
    expr_units:str
    X:np.ndarray
    y:np.ndarray
    
    
    genes:List[str]

    genes_to_idx:Dict[str,int]
    idx_to_id:Dict[int,str]

def write_ICI_DataSet(
    drug:str,
    tissue:str,
    expr_units:str,
    X:np.ndarray,
    y:np.ndarray,
    genes:List[str],
    genes_to_idx:Dict[str,int],
    idx_to_id:Dict[int,str],
    file_outpath:str
    )->None:
    
    dataset = ICIDataSet(
            drug=drug,
            tissue = tissue,
            expr_units = expr_units,
            X=X,
            y=y,
            genes =genes,
            genes_to_idx = genes_to_idx,
            idx_to_id =idx_to_id
            )

    with open(file_outpath, 'wb') as ostream:
        pk.dump(dataset, ostream)


    


DRUG_TISSUE_MAP = {'Atezo':['ALL','BLCA','KIRC'],
                   'Ipi':['ALL','SKCM'],
                   'Nivo':['ALL','KIRC','SKCM'],
                   'Pembro':['ALL','SKCM','STAD']
                    }


DRUG_TARGET_MAP = {'Atezo':'PD-L1','Pembro':'PD1','Nivo':'PD1','Ipi':'CTLA4'}


TARGET_GENE_MAP = {'PD-L1':'CD274', 'PD1':'PDCD1', 'CLTA4':'CTLA4'}


NETPROP_STUDIES = ['Gide','IMVigor']

