
import numpy as np
import slice_graph as sg
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gist_gray'

#Brute force make a 3D matrix, 12 time points long with repeating patterns
A=np.zeros((4,4,1))
B=np.zeros((4,4,1))
A[0,1,0]=1
A[2,3,0]=1
B[0,3,0]=1
B[0,1,0]=1

A_nonflex=np.concatenate([A,B,np.tile(A,5),np.tile(B,2),np.tile(A,3)],2)

#Brute force make a 3D matrix, 12 time points long with lots of different edges
A=np.zeros((4,4,1))
B=np.zeros((4,4,1))
C=np.zeros((4,4,1))
D=np.zeros((4,4,1))
A[0,1,0]=1
A[2,3,0]=1
B[0,3,0]=1
B[0,1,0]=1
C[1,3,0]=1
C[1,2,0]=1
D[0,1,0]=1
D[0,2,0]=1
A_flex=np.concatenate([A,B,B,C,D,A,D,C,B,B,D,C],2)


fig,(ax1,ax2) = plt.subplots(2,1)

edgeList = sg.edgeListFromMatrix(A_nonflex)
ax1 = sg.plot_slice(['node 1','node 2','node 3','node 4'],list(map(str,range(0,12))),ax1,edgeList)
ax1.set_title('A',loc='left')

edgeList = sg.edgeListFromMatrix(A_flex)
ax2 = sg.plot_slice(['node 1','node 2','node 3','node 4'],list(map(str,range(0,12))),ax2,edgeList)
ax2.set_xlabel('time')
ax2.set_title('B',loc='left')
fig.show()

fig.savefig('./figures/fluctuability_example.eps')
fig.savefig('./figures/fluctuability_example.png')
