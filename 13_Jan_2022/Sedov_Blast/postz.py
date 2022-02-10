
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


with open('PlaneXY.pkl', 'rb') as f:
	nXY = pickle.load(f) # the indices of the particles residing in the XY plane at Z = 0.0


j = 94


dirx = './Outputs/'
dirx = './Outputs_U_5.0_h_0.04/'

filez = np.sort(os.listdir(dirx))
with open(dirx + filez[j], 'rb') as f:
	dictx = pickle.load(f)


r = dictx['pos']
rx = r[:, 0]
ry = r[:, 1]
rz = r[:, 2]

plt.figure(figsize = (6, 5))

plt.plot(rx[nXY], ry[nXY], marker = 'D', markersize = 3, markeredgecolor = 'black', markerfacecolor = 'white', linestyle = 'None')

x = y = 0.7

plt.xlim(-x, x)
plt.ylim(-y, y)

plt.show()




