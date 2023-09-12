import sys
from typing import Union, List, Dict, Tuple
import pandas as pd


def load_geneset(
    base_path:str, 
    gs_name:str
    ) -> List[str]:

    


    base_path = base_path + "/" if base_path[-1]!="/" else base_path

    path = "{g}/{n}.txt".format(g = base_path,n= gs_name)
    if gs_name in ['CD8T','TCE','CAF','TAM','GeneBio']:
        with open(path, "r") as istream:
            lines = istream.readlines()
        
        if isinstance(lines, list):
            if len(lines) ==1:
                lines = lines[0].split(":")
                print(lines)
        else:
            lines = lines.split(":")
        
    


    else:
        with open(path, "r") as istream:
            lines = istream.readlines()
        lines = [line.rstrip() for line in lines]
    
    return lines


def fetch_ICI_target_genes():
	drug2gene = {'PD-L1':'CD274', 'PD1':	'PDCD1', 'CTLA4':	'CTLA4'}
	return [x for x in drug2gene.values()]

def harmonize_gene_sets(
    gene_sets:List[List[str]]
    ) -> List[str]:
    result = set(gene_sets[0])
    for i in range(1,len(gene_sets)):
        result = result.union(set(gene_sets[i]))
    return list(result)

def intersect_gene_sets(
    gene_sets:List[List[str]]
    ) -> List[str]:

    result = set(gene_sets[0])
    for i in range(1,len(gene_sets)):
        result = result.intersection(set(gene_sets[i]))

    return list(set(result))

def load_netprop_geneset(
    base_path:str, 
    gs_name:str,
    cut_off:str) -> List[str]:
    

    base_path = base_path + "/" if base_path[-1]!="/" else base_path
    print(base_path)
    print(gs_name)
    path = "{g}{n}.txt".format(g = base_path,n= gs_name)
    gene_scores = pd.read_csv(path, sep = "\t")
    genes = list(gene_scores['geneID'])[:cut_off]
    return genes
    