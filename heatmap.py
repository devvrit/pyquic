import torch
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
torch.set_printoptions(threshold=300)
np.set_printoptions(threshold=300)

S = torch.tensor(np.load("../test_mat_2.pickle", allow_pickle=True)+0.00001*np.eye(512, dtype=np.float64))
print("S dtype: " + str(S.dtype))
# S = torch.tensor(np.load("../test_mat_1.pickle", allow_pickle=True)+0.001*np.eye(1024, dtype=np.float64))

Sinv = torch.tensor(np.linalg.inv(S.double().numpy()))
print("S eigvals: " + str(np.linalg.eigh(S.numpy())[0]))
print("Sinv eigvals: " + str(np.linalg.eigh(Sinv.numpy())[0]))



print("Frob norm of S*Sinv-I is: " + str(torch.norm(torch.matmul(S,Sinv)-torch.eye(S.size(0)))))

matrix = torch.load("X_quic_mat_2.pt") + 1e-32
print("matrix.max() and matrix.min() are: " + str(matrix.abs().max()) + ", " + str(matrix.abs().min()))

matrix = torch.load("X_quic_mat_2.pt") + 1e-32
plt.clf()
# plt.style.use('classic')
fig = plt.figure(figsize=(10,10),dpi=1400)
ax = plt.gca()
im = plt.imshow(torch.log(torch.abs(matrix)), cmap='hot', interpolation='nearest')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=20)
plt.title("heat_map of log(abs(X)), dim="+str(S.size(0))+", lamb=0.05", loc='right')
plt.savefig("X_heatmap_" + str(S.size(0))+"_l_0.05"+ ".png")

matrix = torch.load("../X_quic_mat_2_l_0.5_tle_3_mi_2k.pt") + 1e-32
plt.clf()
fig = plt.figure(figsize=(10,10),dpi=1400)
ax = plt.gca()
im = plt.imshow(torch.log(torch.abs(matrix)), cmap='hot', interpolation='nearest')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=20)
plt.title("heat_map of log(abs(X)), dim="+str(S.size(0))+", lamb=0.5", loc='right')
plt.savefig("X_heatmap_" + str(S.size(0))+"_l_0.5"+ ".png")

matrix = torch.load("../X_quic_mat_2_l_1.0_tle_3_mi_2k.pt")+ 1e-32
plt.clf()
fig = plt.figure(figsize=(10,10),dpi=1400)
ax = plt.gca()
im = plt.imshow(torch.log(torch.abs(matrix)), cmap='hot', interpolation='nearest')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=20)
plt.title("heat_map of log(abs(X)), dim="+str(S.size(0))+", lamb=1.0", loc='right')
plt.savefig("X_heatmap_" + str(S.size(0))+"_l_1.0"+ ".png")



plt.clf()
matrix = Sinv
fig = plt.figure(figsize=(10,10),dpi=1400)
ax = plt.gca()
im = plt.imshow(torch.log(torch.abs(matrix)), cmap='hot', interpolation='nearest')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=20)
plt.title("heat_map of log(abs(Sinv)), dim="+str(S.size(0)), loc='right')
plt.savefig("Sinv_heatmap_" + str(S.size(0))+ "_dim.png")

plt.clf()
matrix = S
fig = plt.figure(figsize=(10,10),dpi=1400)
ax = plt.gca()
im = plt.imshow(torch.log(torch.abs(matrix)), cmap='hot', interpolation='nearest')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=20)
plt.title("heat_map of log(abs(S)), dim="+str(S.size(0)), loc='right')
plt.savefig("S_heatmap_" + str(S.size(0))+ "_dim.png")



# plt.clf()
# fig, (ax1, ax2) = plt.subplots(nrows=2)
# fig.subplots_adjust(wspace=0.01)
# im1 = ax1.imshow(torch.log(torch.abs(matrix)), cmap='hot', interpolation='nearest')
# divider1 = make_axes_locatable(ax1)
# cax1 = divider1.append_axes("right", size="5%", pad=0.1)
# # cax1.yaxis.set_ticks_position("left")
# # cax1.yaxis.tick_left()
# # cax1.xaxis.tick_top()
# # cax1.xaxis.set_label_position('top')
# cbar1 = fig.colorbar(im1, cax=cax1, ax=ax1)
# cbar1.ax.tick_params(labelsize=20)
# # plt.title("heat_map of log(abs(X)), dim="+str(S.size(0)), loc='right')

# ax2.set_axis_off()
# im2 = ax2.imshow(torch.log(torch.abs(Sinv)), cmap='hot', interpolation='nearest')
# divider2 = make_axes_locatable(ax2)
# cax2 = divider.append_axes("right", size="5%", pad=0.1)
# #_ = plt.colorbar(im, cax=cax)
# # cbar = plt.colorbar(im, cax=cax, ax=ax2)
# cbar2 = fig.colorbar(im2, cax=cax2, ax=ax2)
# cbar2.ax.tick_params(labelsize=20)
# # plt.title("heat_map of log(abs(X)), dim="+str(S.size(0)), loc='right')
# # plt.show()
# plt.savefig("some_fig.png")




