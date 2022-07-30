
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time


unitTime_in_Myr =  1.6235259353083704 # Myr


filz = np.sort(glob.glob('./Outputs_10k/*.pkl'))


plt.ion()
fig, ax = plt.subplots(figsize = (8, 6))

kb = ''

for j in range(0, len(filz), 10):

	print('j = ', j)

	with open(filz[j], 'rb') as f:
		data = pickle.load(f)


	r = data['pos']
	h = data['h']
	
	print('h = ', np.sort(h))
	
	#print(r.shape)

	x = r[:, 0]
	y = r[:, 1]
	z = r[:, 2]
	t = data['current_t']
	rho = data['rho']
	#print('rho = ', np.sort(rho))
	
	ax.cla()

	ax.scatter(x, y, s = 0.01, color = 'black')
	xyrange = 1.2
	
	#ax.axis(xmin = -1.2, xmax = 3.2)
	#ax.axis(ymin = -1.2, ymax = 1.5)
	
	ax.axis(xmin = -2.5, xmax = 4.0)
	ax.axis(ymin = -2.5, ymax = 2.5)
	
	
	ax.set_title('t = ' + str(np.round(t*unitTime_in_Myr,4)))
	fig.canvas.flush_events()
	time.sleep(0.01)
	
	#if np.round(t*unitTime_in_Myr,2) > 20000:
	kb =readchar.readkey()
	
	if kb == 'q':
		break

plt.savefig('1111.png')







