import torch
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
torch.set_printoptions(threshold=300)
np.set_printoptions(threshold=300)

# S = torch.tensor(scipy.io.loadmat('cov.mat')['S'])
#S = torch.tensor(np.load("../test_mat_1.pickle", allow_pickle=True))
#X = torch.load("X_quic_.pt")
S = torch.tensor(np.load("../test_mat_2.pickle", allow_pickle=True)+0.00001*np.eye(512, dtype=np.float64))
# S = torch.tensor(np.load("../test_mat_1.pickle", allow_pickle=True)+0.001*np.eye(1024, dtype=np.float64))
X = torch.load("X_quic_mat_2.pt")
# X = torch.load("../X_quic_mat_2_l_1.0_tle_3_mi_2k.pt")
print("S size: " + str(S.size()))
print("X size: " + str(X.size()))
print("total number of entries in S: " + str(S.size(0)**2))
print("num non-zero entries in X: " + str(X.size(0)**2-(X==0).sum()))
print("% of nnzs: " + str(((X!=0).sum())/(S.size(0)**2) ))


evalueS, _ = np.linalg.eigh(S.numpy().astype('d'))
evalueS = -np.sort(-evalueS)
evalueX, _ = np.linalg.eigh(X.numpy())
evalueX = -np.sort(-evalueX)
evalue_invS = -np.sort(-1/evalueS)

print("evalueS: " + str(evalueS))
print("evalueX: " + str(evalueX))
print("1/evalueS: " + str(evalue_invS))
#print("")
#print("diags of S: " + str([S[i,i].item() for i in range(S.size(0))]))
Sinv = torch.tensor(np.linalg.inv(S.double().numpy()))
print("diags of 1/S: " + str(np.array([Sinv[i,i].item() for i in range(Sinv.size(0))])))
print("diags of X: " + str(np.array([X[i,i].item() for i in range(S.size(0))])))
print("")
print("Log scale eigenvals: ")
print("1/S: " + str(torch.log(torch.tensor(evalue_invS))))
print("X: " + str(torch.log(torch.tensor(evalueX))))
print("")
if X.size(0)<3:
	print("X: " + str(X))
	print("Sinv: " + str(Sinv))
	print("S: " + str(S))
#Calculate logDetDiverLoss:
lamb=1.0
a = 0
for e in evalueX:
    a+= -np.log(e)

loss = a+torch.trace(torch.matmul(S.double(),X.double())) + lamb*(torch.sign(X.view(-1))*X.view(-1)).sum()
#loss = -torch.log(torch.det(X.double()))+torch.trace(torch.matmul(S.double(),X.double())) + lamb*(torch.sign(X.view(-1))*X.view(-1)).sum()
print("quic loss: " + str(loss))
loss = -torch.log(torch.det(torch.matmul(X.double(),S.double())))+torch.trace(torch.matmul(S.double(),X.double())) - X.size(0)
#loss = -torch.log(torch.det(X.double()))
loss = -np.log(np.linalg.det(torch.matmul(X.double(), S.double()).numpy().astype('d')))+torch.trace(torch.matmul(S.double(),X.double())) - X.size(0)
print("LogDetDiv: " + str(loss))
print("Frob norm of (X-S^{-1}): " + str(torch.norm(X-Sinv)))
print("XS-I frob norm: " + str(torch.norm(torch.matmul(X.double(),S.double())-torch.eye(X.size(0)))))

'''
plt.plot([i for i in range(len(evalueX))], torch.log(torch.tensor(evalue_invS)), marker=".", markersize=3)
# plt.xlabel("eigenvalues")
plt.ylabel("log(eigenval)")
plt.title("log-scale eigenvalues of inverse coviance matrix, dim="+str(len(evalueX)))
plt.savefig("invCov.png")

plt.clf()

plt.plot([i for i in range(len(evalueX))], torch.log(torch.tensor(evalueX)), marker=".", markersize=5)
# plt.xlabel("eigenvalues")
plt.ylabel("log(eigenval)")
plt.title("log-scale eigenvalues of inv-cov matrix by QUIC, dim="+str(len(evalueX)))
plt.savefig("quicInvCov.png")

plt.clf()

plt.plot([i for i in range(len(evalueX))], torch.log(torch.tensor(evalue_invS)), "r+-", markersize=3)
plt.plot([i for i in range(len(evalueX))], torch.log(torch.tensor(evalueX)), "go-", markersize=3)
plt.ylabel("log(eigenval)")
plt.title("log-scale eigenvalues of inv-cov matrices, dim="+str(len(evalueX)))
plt.legend(["S^{-1}", "X"])
plt.savefig("compare_eigenvals_"+str(len(evalueX))+".png")

plt.clf()
'''