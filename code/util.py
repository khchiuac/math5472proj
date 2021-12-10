# -----------------------------------------------------------
# utility functions for MATH 5472 project
#
# December 2021
# -----------------------------------------------------------
import numpy as np
from numpy.random import *
import time
from copy import *
import mosek
import cvxpy as cp
from scipy.stats import norm
from scipy.linalg import qr
from scipy.sparse.linalg import svds
from scipy.optimize import minimize
from sklearn.datasets import make_blobs
import json
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

seed(0)

def em_update(L, x):
    n, m = L.shape
    if x.shape[0]!= m:
        print("Dimension of x is wrong")
        return
    
    phi = np.zeros((n, m))
    for j in range(n):
        denominator = np.matmul(L, x)
        for k in range(m):
            phi[j, k] = L[j, k]*x[k]
    
    return phi.mean(axis=0)

def active_set(g, H, W_0, epsilon, step_limit=1, method="Powell"):
    def active_obj1(q):
        return np.add(0.5*np.matmul(np.matmul(np.transpose(q), H), q), np.matmul(np.transpose(q), b))
    def active_con1(q):
        for i in range(m):
            if indicator[i]==1 and q[i]!=0:
                return False
        return True

    m = g.shape[0]
    k = m-len(W_0)
    y = np.zeros(m)
    W = W_0
    indicator = np.zeros(m)
    for i in range(m):
        y[i] = 0 if (i+1) in W_0 else 1./k
        indicator[i] = 1 if (i+1) in W_0 else 0 # IN:1

    for l in range(step_limit):
        b = np.add(np.matmul(H, y), 2*g)+1
        fun = active_obj1
        cons = [{'type':'eq', 'fun': active_con1}]
        res = minimize(fun, np.zeros(m), method=method, constraints=cons)
        q_l = res.x
        alpha_l = 1

        if max(abs(q_l))<=0:
            if min(np.multiply(b, indicator))>=-epsilon:
                return
            fun = lambda i: b[int(i)]
            cons = ({'type': 'eq', 'fun': lambda i: indicator[int(i)]-1})
            res = minimize(fun, 0, method=method, constraints=cons)
            W -= set({int(res.x)+1})
        else:
            fun = lambda i: -y[int(i)]/q_l[int(i)]
            cons = ({'type': 'eq', 'fun': lambda i: -indicator[int(i)]}, {'type': 'ineq', 'fun': -q_l[int(i)]})
            res = minimize(fun, 0, method=method, constraints=cons)
            alpha_l = -y[int(res.x)]/q_l[int(res.x)]
            if alpha_l<1:
                W.add(int(res.x)+1)
        y = y+alpha_l*q_l
    return y

def tilde_f(L, x):
    return -np.sum(np.log(L@x))/n+np.sum(x)