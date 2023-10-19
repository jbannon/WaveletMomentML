import numpy as np
from sklearn.utils import check_array, check_random_state
from sklearn.base import BaseEstimator, TransformerMixin
import sys 
import pickle as pk 
import networkx as nx

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import pandas as pd





class WaveletMomentTransform():
	def __init__(self,
		numScales:int,
		maxMoment:int,
		adjacency_matrix:np.ndarray,
		central:bool):

		assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "adjacency matrix must be square"
		# assert (adjacency_matrix == adjacency_matrix.T).all(), "adjacency_matrix must be symmetric"


		self.numScales = numScales
		self.maxMoment = maxMoment
		self.adjacency_matrix = adjacency_matrix.copy()
		self.central = central
		self.H = None





	def computeTransform(self,
		X:np.ndarray,
		) -> np.ndarray:
		
		S = X.shape[0]
		N = X.shape[1]
		

		new_X = self.H @ X.T
		
		exponents = np.arange(1,self.maxMoment+1)
	
		X_res= np.array([])

		for s in range(S):
			sample_transforms = new_X[:,:,s]
			sample_coeffs = np.array([])

			sample_transforms = np.abs(sample_transforms)
			
			for exp in exponents:

				if exp == 1 and self.central:

					coeffs = np.mean(sample_transforms,axis=1)

				elif exp ==2 and self.central: 

					coeffs = np.var(sample_transforms,axis =1)

				elif exp > 2 and self.central:

					mu = np.mean(sample_transforms, axis=1, keepdims = True)
					sigma = np.std(sample_transforms, axis=1, keepdims = True)
					coeffs = (sample_transforms - mu)/sigma
					coeffs = np.sum(np.power(coeffs,exp),axis=1)

				else:

					coeffs = np.sum(np.power(np.abs(sample_transforms),exp),axis=1)
				
				sample_coeffs = np.hstack([sample_coeffs, coeffs]) if sample_coeffs.size else coeffs

				
			X_res = np.vstack( [X_res, sample_coeffs]) if X_res.size else sample_coeffs
		
		
		if np.isnan(X_res).any():
			print("self reporting an issue with scale {J} and max moment {p}".format(J=self.numScales, p=self.maxMoment))
			sys.exit(1)
		return X_res
		


class DiffusionWMT(WaveletMomentTransform):
	def __init__(self,
		numScales:int,
		maxMoment:int,
		adjacency_matrix:np.ndarray,
		central:bool = True):

		super().__init__(numScales, maxMoment, adjacency_matrix,central)


		
		N  = self.adjacency_matrix.shape[0]
		
		max_J = self.numScales
		
		D_invsqrt =  np.diag(1/np.sqrt(np.sum(self.adjacency_matrix,axis=1)))
		
		A = D_invsqrt @ self.adjacency_matrix @ D_invsqrt
		
	
		T = 0.5*(np.eye(A.shape[0])+A)
		
		H = (np.eye(N) - T).reshape(1, N, N)

		

			

		for j in range(1,max_J):
			# print("computing {j}".format(j=j))
			new_wavelet = np.linalg.matrix_power(T,2**(j-1)) - np.linalg.matrix_power(T,2**j)
			H = np.concatenate((H,new_wavelet.reshape(1,N,N)),axis=0) if H.size else new_wavelet.reshape(1,N,N)
		
		
		self.H = H 


class DiffusionWaveletExpansion():
	def __init__(self,
		numScales:int,
		adjacency_matrix:np.ndarray):

		assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "adjacency matrix must be square"
		# assert (adjacency_matrix == adjacency_matrix.T).all(), "adjacency_matrix must be symmetric"


		self.numScales = numScales
		self.adjacency_matrix = adjacency_matrix.copy()
		
		
		N  = self.adjacency_matrix.shape[0]
		
		max_J = self.numScales
		
		D_invsqrt =  np.diag(1/np.sqrt(np.sum(self.adjacency_matrix,axis=1)))
		
		A = D_invsqrt @ self.adjacency_matrix @ D_invsqrt
		
	
		T = 0.5*(np.eye(A.shape[0])+A)
		
		H = (np.eye(N) - T).reshape(1, N, N)

		
		for j in range(1,max_J):
			# print("computing {j}".format(j=j))
			new_wavelet = np.linalg.matrix_power(T,2**(j-1)) - np.linalg.matrix_power(T,2**j)
			H = np.concatenate((H,new_wavelet.reshape(1,N,N)),axis=0) if H.size else new_wavelet.reshape(1,N,N)
		
		
		self.H = H 





	def computeTransform(self,
		X:np.ndarray,
		) -> np.ndarray:
		
		S = X.shape[0]
		N = X.shape[1]
		

		new_X = self.H @ X.T
		X_res= np.array([])
		
		for i in range(new_X.shape[0]):
			X_res= np.hstack( [X_res, new_X[i,:,:].T]) if X_res.size else new_X[i,:,:].T
		
		atoms = np.array([])
		for i in range(self.H.shape[0]):
			atoms = np.vstack( [atoms, self.H[i,:,:]]) if atoms.size else self.H[i,:,:]

		
		
		
		
		return X_res, atoms 

		