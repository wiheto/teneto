
import numpy as np
import slice_graph as sg
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gist_gray'


#Creat the 3d matrix
A=np.zeros((3,3,3))
A[0,1,:]=1
A[0,2,1:]=1
A[1,2,2]=1

edgeList = sg.edgeListFromMatrix(A)

fig, ax = plt.subplots(1,1)
ax = sg.plot_slice(['Ashley','Blake','Casey'],['2014','2015','2016'],ax,edgeList)
ax.set_xlabel('time (years)')
fig.savefig('./figures/friendexample.png')
fig.savefig('./figures/friendexample.eps')
