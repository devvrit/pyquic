import torch
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
torch.set_printoptions(threshold=300)

S = torch.tensor(scipy.io.loadmat('cov.mat')['S'])
X = torch.load("X_quic.pt")
print("S size: " + str(S.size()))
print("total number of entries in S: " + str(S.size(0)**2))
print("num non-zero entries in X: " + str(X.size(0)**2-(X==0).sum()))


evalueS, _ = np.linalg.eig(S.numpy())
evalueS = -np.sort(-evalueS)
evalueX, _ = np.linalg.eig(X.numpy())
evalueX = -np.sort(-evalueX)
evalue_invS = -np.sort(-1/evalueS)

print("evalueS: " + str(evalueS))
print("evalueX: " + str(evalueX))
print("1/evalueS: " + str(evalue_invS))
print("")
print("diags of S: " + str([S[i,i].item() for i in range(S.size(0))]))
Sinv = torch.inverse(S)
print("diags of 1/S: " + str([Sinv[i,i].item() for i in range(Sinv.size(0))]))
print("diags of X: " + str([X[i,i].item() for i in range(S.size(0))]))
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
lamb=0.5
# torch.trace(torch.matmul(S,X))
# -torch.log(torch.det(X))
diag = [i for i in range(X.size(0))]
print("(torch.sign(X.view(-1))*X.view(-1)): " + str((torch.sign(X[diag,diag].view(-1))*X[diag,diag].view(-1))))
loss = -torch.log(torch.det(X.double()))+torch.trace(torch.matmul(S.double(),X.double())) + lamb*(torch.sign(X.view(-1))*X.view(-1)).sum()
print("loss: " + str(loss))

plt.plot([i for i in range(len(evalueX))], torch.log(torch.tensor(evalue_invS)), marker=".", markersize=5)
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