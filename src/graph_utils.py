import numpy as np 
import sys 
import networkx as nx 
from typing import Union, List, Dict, Tuple
import time	

def harmonize_graph_and_geneset(
	G:nx.Graph,
	gene_set:List[str]
	) -> nx.Graph:
	

	common_genes = list(set(G.nodes).intersection(set(gene_set)))
	

	H = G.subgraph(common_genes)
	
	if not nx.is_connected(H):
		LCC_genes = max(nx.connected_components(H), key=len)
		H = H.subgraph(LCC_genes)

	return H


def make_diffusion_operator(
	G:nx.Graph,
	)-> np.ndarray:
	
	W = nx.adjacency_matrix(G).todense()
	D_sqrt =  np.diag(np.array( [np.sqrt(G.degree[n]) for n in G.nodes()]))
	s = time.time()
	A = D_sqrt @ W @ D_sqrt
	e = time.time()
	T = 0.5*(np.eye(A.shape[0])+A)
	
	return T
	