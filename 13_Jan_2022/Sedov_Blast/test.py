
import numpy as np
import matplotlib.pyplot as plt



stp = 1. / 32.

xL = np.arange(-.50, 0.0, stp)
xR = np.arange(0, 0.5, stp)
X = np.concatenate((xL, xR))

Y = X.copy()
Z = X.copy()

#print(x)
#print(len(x))

N = len(X)**3

r = np.zeros((N, 3))

print(r.shape)

i = 0

for x in X:

	for y in Y:

		for z in Z:
		
			r[i, 0] = x
			r[i, 1] = y
			r[i, 2] = z
			
			i += 1



plt.scatter(r[:, 0], r[:, 1], s = 1)
plt.show()












