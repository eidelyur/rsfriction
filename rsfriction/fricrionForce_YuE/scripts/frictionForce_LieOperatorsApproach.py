# from __future__ import division

#-------------------------------------
#
#        Started at 07/25/2017 (YuE)
# 
#-------------------------------------

import os, sys
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib as mpl

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

import scipy.integrate as integrate
from scipy.integrate import quad, nquad, dblquad

from scipy.constants import pi
from scipy.constants import speed_of_light as clight
from scipy.constants import epsilon_0 as eps0
from scipy.constants import mu_0 as mu0
from scipy.constants import elementary_charge as qe
from scipy.constants import electron_mass as me
from scipy.constants import proton_mass as mp
from scipy.constants import Boltzmann as kB

#--------------------------------------------------------------------------------------
#
# Main idea:
#
# F=2 \cdot \pi \cdot n_e \cdot \
#     \int U \cdot \vec {deltaP_e(\vec U)} f(\vec {V_e}) \vec {dV_e} \rho d \rho,
#
# where \vec {U} = \vec {V_i} - \vec {V_e} - relative velocity of the ion and electron,
#       \vec deltaP_e(\vec {U}) - momentum ransfer from ipn to electron due to collision,
#       f(\vec {V_e}) - electron velocity distribution function,
#       \rho - collision impact parameter;
#
# Value of \vec {deltaP_e(\vec U)} is calculated using "Magnus expand approach" (MEA)
#
#--------------------------------------------------------------------------------------

eVtoErg=1.602e-12                # energy from EV to erg
#
# Initial parameters:
#
Z_ion = qe*2.997e+9              # charge of ion (proton), CGSE units of the charge
M_ion = mp*1.e+3                 # mass of ion (proton), g
q_elec = qe*2.997e+9             # charge of electron, CGSE units of the charge
m_elec = me*1.e+3                # mass of electron, g
L_intrxn = 50.e-4                # length interaraction, cm 
B_mag = 1000.                    # magnetic field, Gs
Temp_eTran = 0.5                 # transversal temperature of electrons, eV
Temp_eLong = 2.e-4               # longitudinal temperature of electrons, eV
numb_e = 1000                    # number of electrons
numb_p = 50                      # number of protons

a_eBeam = 0.5                    # cm
n_eBeam = 1.e+9                  # cm^-3
"""
Angular frequency of Larmor rotations for electron:
"""
def omega_Larmor(mass, B):
    return (qe*2.997e+9) * B / (mass*clight*1.e+2)

#
# Derived quantities:
#
tempRatio=Temp_eLong/Temp_eTran
velRatio=np.sqrt(tempRatio)
print 'tempRatio = %e, velRatio = %e' % (tempRatio,velRatio)


Omega_e = omega_Larmor(m_elec, B_mag)                    # rad/sec 
T_larm = (2 * pi) / Omega_e                              # sec
print 'omega_Larmo r= %e rad/sec, T_larm = %e sec' % (Omega_e,T_larm)

rmsV_eTran = np.sqrt(2.*Temp_eTran*eVtoErg/m_elec)       # cm/sec
rmsV_eLong = np.sqrt(2.*Temp_eLong*eVtoErg/m_elec)       # cm/sec
print 'rmsV_eTran = %e cm/sec, rmsV_eLong = %e cm/sec' % (rmsV_eTran,rmsV_eLong)

ro_larm = rmsV_eTran/Omega_e                             # cm
print 'ro_larm = %e cm' % ro_larm

omega_e=np.sqrt(4*pi*n_eBeam*q_elec**2/m_elec)           # rad/sec
print 'omega_e = %e rad/sec' % omega_e

z_ion=np.zeros((6,numb_p))          # vector: x_e,px_e,y_e,py_e,z_e,pz_e for each electron
z_elec=np.zeros((6,numb_e))         # vector: x_i,px_i,y_i,py_i,z_i,pz_i for each proton

#
# Initial uniform distribution of the electron's impact parameter:
#
impctPar = np.random.uniform(high=a_eBeam,size=numb_e)

# Verifying of distribution:
plt.figure(10)
plt.hist(impctPar,bins=30)
plt.xlabel('Impact parameters, cm',color='m',fontsize=16)
plt.ylabel('Particles',color='m',fontsize=16)
plt.title('Electron''s Initial Distribution',color='m',fontsize=16)
plt.grid(True)

#
# Initial uniform distribution of the electron's in cross section:
#
phi=np.random.uniform(high=2*pi,size=numb_e)
for i in range(numb_e):
   z_elec[0,i]=impctPar[i]*math.cos(phi[i])
   z_elec[2,i]=impctPar[i]*math.sin(phi[i])

# Verifying of distribution:
plt.figure(20)
plt.plot(z_elec[0,:],z_elec[2,:],'.r',linewidth=2)
plt.xlabel('$x_e$, cm',color='m',fontsize=16)
plt.ylabel('$y_e$, cm',color='m',fontsize=16)
plt.title('Electron''s Initial Distribution',color='m',fontsize=16)
plt.xlim([np.min(z_elec[0,:]),np.max(z_elec[0,:])])
plt.ylim([np.min(z_elec[2,:]),np.max(z_elec[2,:])])
plt.grid(True)
plt.axes().set_aspect('equal')

#
# Initial gaussian distributions of the relative transverse electron's velocities 
#
z_elec[1,:]=np.random.normal(scale=1.0,size=numb_e)
z_elec[3,:]=np.random.normal(scale=1.0,size=numb_e)

# Verifying of distributions:
stdVex=z_elec[1,:].std()
stdVey=z_elec[3,:].std()
print 'stdVex = %e (must be 1.0), stdVey = %e (must be 1.0)' % (stdVex,stdVey)

plt.figure(30)
vel_hist=plt.hist(z_elec[1,:],bins=30)
plt.xlabel('$V_{ex} / V_{e\perp}$',color='m',fontsize=16)
plt.ylabel('Particles',color='m',fontsize=16)
plt.ylim([0,1.1*np.max(vel_hist[0])])
plt.title(('Electron''s Initial Distribution: $V_{rms}$ = %6.4f' % stdVex), \
          color='m',fontsize=16)
plt.text(0.,1.025*np.max(vel_hist[0]),('Initial: $V_{rms}$ = %6.4f' % 1.), \
         color='m',fontsize=16,ha='center')	  
plt.grid(True)

plt.figure(40)
vel_hist=plt.hist(z_elec[3,:],bins=30)
plt.xlabel('$V_{ey} / V_{e\perp}$',color='m',fontsize=16)
plt.ylabel('Particles',color='m',fontsize=16)
plt.ylim([0,1.1*np.max(vel_hist[0])])
plt.title(('Electron''s Initial Distribution: $V_{rms}$ = %6.4f' % stdVey), \
          color='m',fontsize=16)
plt.text(0.,1.025*np.max(vel_hist[0]),('Initial: $V_{rms}$ = %6.4f' % 1.), \
         color='m',fontsize=16,ha='center')	  
plt.grid(True)

#
# Initial gaussian distribution of the relative longitudinal electron's velocities 
#
z_elec[5,:]=np.random.normal(scale=1.0*velRatio,size=numb_e)

# Verifying of distribution:
stdVez=z_elec[5,:].std()
print 'stdVexz = %e (must be 1.0)' % stdVez

plt.figure(50)
vel_hist=plt.hist(z_elec[5,:],bins=30)
plt.xlabel('$V_{ez} / V_{e\perp}$',color='m',fontsize=16)
plt.ylabel('Particles',color='m',fontsize=16)
plt.ylim([0,1.1*np.max(vel_hist[0])])
plt.title(('Electron''s Initial Distribution: $V_{rms}$ = %6.4f' % stdVez), \
          color='m',fontsize=16)
plt.text(0.,1.025*np.max(vel_hist[0]),('Initial: $V_{rms}$ = %6.4f' % velRatio), \
         color='m',fontsize=16,ha='center')	  
plt.grid(True)


N_gyro = 100                   # a somewhat arbitrary choice, range [100, 160] 

plt.show()   

sys.exit()   

# plt.text(-0.275e+8,1.025*np.max(vel_hist[0]),('Initial: $V_{rms}$ = %8.3e cm/sec' % rmsV_eTran), \
#          color='m',fontsize=16)	  
# plt.xlabel('$V_{e\perp} / \sqrt {2 \cdot T_{e\perp}/m}$',color='m',fontsize=16)
