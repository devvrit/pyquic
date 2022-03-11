import random
from scipy.io import savemat
import torch
import numpy as np
torch.set_printoptions(linewidth=200)

def compute_covariance(Y):
	Y = Y.clone()
	Y_mean = torch.mean(Y, dim=-1)
	Y = Y - Y_mean.view(-1,1)
	Z = torch.zeros(Y.size(0),Y.size(0))
	for i in range(Y.size(1)):
		y = Y[:,i].view(-1,1)
		Z+= torch.matmul(y,y.T)
	Z = Z/(Y.size(1)-1)
	return Z


def perturb(y):
	y = y.clone()
	# return y
	for j in range(y.size(0)):
		y[j]=y[j]*((1+random.uniform(-1, 1)))
	return y

k=10
r=5
x = []

# i=r
# while i>=(-r):
# 	x.append(10**i)
# 	i-=1


i=r/2
while i>=(-r/2):
	x.append(10**i)
	i-=r/(2*k+1)
# x = torch.tensor([10**i for i in reversed(range(-r/2,(r/2)+0.00000001), r/(2*k+1))])
print("x: " + str(x))
x = torch.tensor(x)
n=100000

X = []
for i in range(n):
	X.append(perturb(x).view(-1,1))

X = torch.cat(X, dim=1)
print("X size is: " + str(X.size()))
print("X is: " + str(X))
print("")


# cov = torch.cov(X)
cov = torch.matmul(X,X.T)/X.size(1)
print("Covariance matrix size: " + str(cov.size()))
print("cov: " + str(cov[:2,:]))

print("")
evalue, evect = np.linalg.eig(cov.numpy())
print("evalue: " + str(np.sort(evalue)))

mdict = {'S':cov.cpu().numpy()}
savemat("cov.mat", mdict)




#######################################
'''
cov_custom = torch.tensor([[1+random.uniform(0, 1), 1.0],[1.0,1.0]])
print("cov: " + str(cov_custom))
# X = []
# for i in range(10000):
# 	X.append(torch.tensor([[1+random.uniform(-1, 1), 1.0],[1.0,1.0]]))
# X = torch.append

evalue, evect = np.linalg.eig(cov_custom.numpy())
print("evalue: " + str(np.sort(evalue)))

mdict = {'S':cov_custom.cpu().numpy()}
savemat("cov.mat", mdict)
'''