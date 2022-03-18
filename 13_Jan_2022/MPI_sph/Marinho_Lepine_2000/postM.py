
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time

#--- Scaling From Marinho et al. (2000)-----
M_sun = 1.989e33 # gram
UnitRadius_in_cm = 3.086e18 # == 1 pc
UnitTime_in_s = 3.16e13 # == 1Myr
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2

UnitMass_in_g = UnitRadius_in_cm**3 / grav_const_in_cgs / UnitTime_in_s**2
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3
#-------------------------------------------
mH = 1.6726e-24 # gram

dirx = './Outputs/'

filz = np.sort(glob.glob(dirx+'*.pkl'))


plt.ion()
fig, ax = plt.subplots(figsize = (8, 8))

t = 0.0

kb = ''

for j in range(0, len(filz), 1):

	with open(filz[j], 'rb') as f:
		res = pickle.load(f)
	
	r = res['pos']
	u = res['u']
	rho = res['rho']
	nH = rho * UnitDensity_in_cgs / mH
	dt = res['dt']
	
	print('u = ', np.min(u), np.max(u))
	print('rho = ', np.sort(rho))
	print()
	
	x = r[:, 0]
	y = r[:, 1]
	z = r[:, 2]
	
	nn = np.where(nH > 3000.)[0]

	xH = x[nn]
	yH = y[nn]
	zH = z[nn]

	t += dt

	ax.cla()	

	#ax.scatter(y, z, s = 0.05, color = 'black')
	ax.scatter(yH, zH, s = 0.20, color = 'lime')

	ax.axis(xmin = -3, xmax = 3)
	ax.axis(ymin = -3, ymax = 3)
	
	#ax.axis(xmin = -24, xmax = 24)
	#ax.axis(ymin = -24, ymax = 24)
	
	ax.set_title('t = ' + str(t))

	fig.canvas.flush_events()
	time.sleep(0.01)

	if t > 3.80:
		kb = readchar.readkey()
	
	if kb == 'q':
		break

	


