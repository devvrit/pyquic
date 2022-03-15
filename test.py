"""
Testing and demonstrating the usage of the python
bindings for QUIC[1]. For the original code (C++ and
bindings for MatLab and R), see
http://www.cs.utexas.edu/~sustik/QUIC/.

[1] Sparse Inverse Covariance Matrix Estimation Using Quadratic Approximation, 
Cho-Jui Hsieh, Matyas A. Sustik, Inderjit S. Dhillon, Pradeep Ravikumar,
Proc. NIPS 2011
"""
from scipy.io import loadmat
import numpy as np
from numpy.testing import assert_allclose
from os.path import exists
import torch

import py_quicx

assert exists("ER_692.mat"), "ER_692.mat missing.\n"+\
        "Obtain this file either from the QUIC website in the MEX "+\
        "package archive[1], or from [2].\n" +\
        "[1] http://www.cs.utexas.edu/~sustik/QUIC/\n" +\
        "[2] http://www.math.nus.edu.sg/~mattohkc/Covsel-0.zip"

# A 692 x 692 empirical covariance matrix
#S = loadmat("ER_692.mat")['S']
#S = loadmat("cov.mat")['S']
#S = np.load("../test_mat_2.pickle", allow_pickle=True) + 0.00001*np.eye(512, dtype=np.float64)
S = np.load("../test_mat_1.pickle", allow_pickle=True) + 0.001*np.eye(1024, dtype=np.float64)
print(torch.tensor(np.linalg.eigh(S)[0]).sort()[0])
newS = np.zeros(S.shape)
print(type(S))
print(S.shape)
# print("S: " + str(S))
newS[:] = S
# for i in range(len(newS)):
	# newS[i,i]=S[i,i]
	# newS[i,i]=1.0

# print(newS)
# print(S)

# Run in "default" mode
X, W, opt, cputime, iters, dGap = py_quicx.quic(S=newS, L=0.01,\
        tol=1e-3, max_iter=2000, msg=2)
print(X.shape)
torch.save(torch.tensor(X), "X_quic_mat_1_l_0.01_tle_3_mi_2k.pt")
# test = 923.1042
# assert_allclose(opt, test)
assert 1==2

# Run in path mode
print("")
path = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
XP, WP, optP, cputimeP, iterP, dGapP = py_quicx.quic(S=newS, L=1.0, 
        mode="path", path=path, tol=1e-16, max_iter=100, msg=2)
# test = np.array([1171.6578, 1136.1222, 1097.9438, 
#     1053.0555, 995.6587, 923.1042])
# assert_allclose(optP, test)

# Run in trace mode
print("")
XT, WT, optT, cputimeT, iterT, dGapT = py_quicx.quic(S=newS, L=0.5, \
        mode="trace", tol=1e-16, max_iter=iters, msg=1)
# test = np.array([993.2862, 965.4918, 927.3593, 923.3665, 923.1369,
#     923.1083, 923.1045, 923.1042, 923.1042, 923.1042, 923.1042])
# assert_allclose(optT, test, rtol=1e-2)

print("\nBasic tests passed.")
