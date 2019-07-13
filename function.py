
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:58:33 2019

@author: chendan
"""
import networkx as nx
import numpy as np
from scipy import linalg

#Compute the adjacency matrix and the Laplace matrix
def cal_AandL(G):
    # Gets a sparse storage format for the network's adjacency matrix
    edgelist = nx.adjacency_matrix(G) 
    # Get the adjacency matrix of network G (in two-dimensional array form)
    A = np.array(edgelist.todense()) 
    A_2 = np.dot(A, A)
    # Diagonal matrix consisting of degrees of nodes
    D = np.diag(np.diag(A_2)) 
    L = D - A
    return A, L

#========Calculation of density matrix=======
def cal_density_matrix1(H):
    eig = linalg.eigvals(H)
    Z = np.sum(eig)
    p = (H/Z).real
    return p

def cal_density_matrix2(beta,H):
    eig = linalg.eigvals(H)
    Z = np.sum(np.exp(-beta*eig))
    p = (linalg.expm(-beta*H)/Z).real
    return p
#============================================

#=======Calculation of spectral entropy======
def cal_entropy1(H):  
    n = np.shape(H)[0]
    nom = np.log2(n*1.0)
    eig = linalg.eigvals(H).real
    Z = np.sum(eig)
    S1 = np.log2(Z)
    S2 = 0.0
    # Due to the calculation accuracy problem, the
    # condition in Numpy is used to brush the elements
    # less than or equal to 0 in eig and replace it with 1.
    # In fact, the Laplace matrix has no eigenvalues less than 0.
    eig[eig <= 0] = 1
    '''
    for i in range(len(eig)):
        if eig[i] > 0:
            S2 = S2 + eig[i]*np.log(eig[i])
    '''
    S2 = np.sum(eig*np.log(eig))
    S = S1 - S2/(Z*np.log(2.0))
    return S/nom

def cal_entropy2(beta,H):  
    n = np.shape(H)[0]
    nom = np.log2(n*1.0)
    eig = linalg.eigvals(H).real
    Z = np.sum(np.exp(-beta*eig), dtype=np.float64)
    S1 = np.log2(Z)
    S2 = (np.sum(beta*eig*np.exp(-beta*eig), dtype=np.float64))/(Z*np.log(2.0))
    S = S1 + S2
    return S/nom
#============================================

#==Calculation of Jensen-Shannon divergence==
def cal_JSD(p1,p2):
    rho = (p1 + p2)/2
    eig1 = linalg.eigvals(p1).real
    eig2 = linalg.eigvals(p2).real
    eig = linalg.eigvals(rho).real
    sp = 0.0
    eig1[eig1 <= 0] = 1
    eig2[eig2 <= 0] = 1
    eig[eig <= 0] = 1
    '''
    for i in range(len(eig)):
        if eig[i] > 0:
            sp = sp + eig[i]*np.log2(eig[i])
    '''
    s1 = -np.sum(eig1*np.log2(eig1))
    s2 = -np.sum(eig2*np.log2(eig2))
    sp = -np.sum(eig*np.log2(eig))
    DJS = sp - (s1 + s2)/2
    return DJS
#============================================