# from __future__ import division

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

from scipy.constants import pi
from scipy.constants import speed_of_light as clight
from scipy.constants import epsilon_0 as eps0
from scipy.constants import mu_0 as mu0
from scipy.constants import elementary_charge as qe
from scipy.constants import electron_mass as me
from scipy.constants import proton_mass as mp
from scipy.constants import Boltzmann as kB

fourPiEps0 = 4 * pi * eps0
invFourPiEps0 = 1 / fourPiEps0
eVtoErg=1.602177e-12

"""
print '\nStart...\n'
print 'Elementary charge (C) = %e' % qe
print 'Electron mass (kg) = %e' % me
print 'Proton mass (kg) = %e' % mp
print 'epsilon_0 (F/m)= %e' % eps0
print 'clight (m/s) = %e' % clight
print 'mu0 (N/A^2)= %e' % mu0
print 'Boltzmann (J/K) = %e' % kB 
print 'fourPiEps0 (F/m) = %e' % fourPiEps0
print 'invFourPiEps0 (m/F) = %e' % invFourPiEps0
"""

# Elementary charge (C) = 1.602177e-19
# Electron mass (kg) = 9.109383e-31
# Proton mass (kg) = 1.672622e-27
# epsilon_0 (F/m)= 8.854188e-12
# clight (m/s) = 2.997925e+08
# mu0 (N/A^2)= 1.256637e-06
# Boltzmann (J/K) = 1.380649e-23
# fourPiEps0 (F/m) = 1.112650e-10
# invFourPiEps0 (m/F) = 8.987552e+09

# Range for electron Larmour radius (m) depend on nagnetic 
# field and transverse electron temperature
#
# Magnetic field B_f (T) and 
# cyclic electron cyclotron frequency (rad/s):
minB=.05   # T
maxB=.4    # T
numbB=50
B_f=np.zeros(numbB)
omegaE=np.zeros(numbB)
for i in range(numbB):
   B_f[i]=minB+(maxB-minB)/(numbB-1)*i
   omegaE[i]=qe*B_f[i]/me
print 'B_f, omegaE=',B_f, omegaE

plt.figure(10)
plt.plot(B_f,omegaE*1e-9,'-b',linewidth=3)
plt.xlim([minB,maxB])
plt.title('Electron Larmour Frequency', color='m',fontsize=20)
plt.xlabel('B (T)',color='m',fontsize=16)
plt.ylabel('$\Omega_e$ (GHz)',color='m',fontsize=16)
plt.grid(True)

# Transverse temperature T_tr (eV) and
# transverse electron velocity V_tr (cm/s):
minT_tr=.01   # eV
maxT_tr=2.0   # eV
numbT=50
T_tr=np.zeros(numbT)
V_tr=np.zeros(numbT)
for i in range(numbT):
   T_tr[i]=minT_tr+(maxT_tr-minT_tr)/(numbT-1)*i
   V_tr[i]=np.sqrt(2*T_tr[i]*eVtoErg/(me*1e3))
print 'T_tr, V_tr=',T_tr,V_tr

plt.figure(20)
plt.plot(T_tr,V_tr*1e-7,'-b',linewidth=3)
plt.xlim([minT_tr,maxT_tr])
plt.title('Electron Transverse Velocity', color='m',fontsize=20)
plt.xlabel('Electron Transverse Temperature $T_\perp$ (eV)',color='m',fontsize=16)
plt.ylabel('$10^{-5} \cdot V_\perp$ (m/s)',color='m',fontsize=16)
plt.grid(True)

# Electron Larmor radius ro_L (mkm):

ro_L=np.zeros((numbB,numbT))
for i in range(numbB):
   for k in range(numbT):
      ro_L[i,k]=V_tr[k]/omegaE[i]*1e4
      
print 'ro_L for minB=',ro_L[0,:]      
print 'ro_L for maxB=',ro_L[numbB-1,:]      
print 'ro_L for minT_tr=',ro_L[:,0]      
print 'ro_L for maxT_tr=',ro_L[:,numbT-1]      

X,Y=np.meshgrid(B_f,T_tr)      
fig100=plt.figure(100)
ax100=fig100.gca(projection='3d')
surf=ax100.plot_surface(X,Y,ro_L,cmap=cm.coolwarm,linewidth=0,antialiased=False)
# plt.xlim([minB,maxB])
plt.title('Electron Larmour Radius', color='m',fontsize=20)
plt.xlabel('B (T)',color='m',fontsize=16)
plt.ylabel('$T_\perp$ (eV)',color='m',fontsize=16)
ax100.set_zlabel('$R_L (\mu m)$',color='m',fontsize=16)
fig100.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()
      

#
# Function psi(a) from Bruhwiler et al ""Direct simulation... "
#
amax=250.
n_a=5000
a=np.zeros(n_a)
psi=np.zeros(n_a)
psi_b=np.zeros(n_a)
dv1=np.zeros(n_a)
dv2=np.zeros(n_a)
for i in range(n_a):
   a[i]=1.+amax/(n_a-1)*i
   a2=a[i]/np.sqrt(2)
   psi[i]=math.erf(np.sqrt(a[i]))-2.*np.sqrt(a[i]/pi)*math.exp(-a[i])
   psi_b[i]=math.erf(a2)-2.*a2*np.sqrt(1./pi)*math.exp(-a2**2)
   dv1[i]=math.fabs(1/a[i]**2*math.log(100*a[i]**3)*psi_b[i])
   dv2[i]=math.fabs(1/a[i]**2*math.log(100*a[i]**3))

print 'a=',a[1:50]   
print 'psi=',psi[1:50]   
print 'dv=',dv1[1:50]   

fig300=plt.figure(300)
plt.loglog(a,psi,'-xr')

fig400=plt.figure(400)
plt.loglog(a,dv1,'-r')
plt.loglog(a,dv2,'-b')


# plt.show()   



#
# Verifying some items from Parchomchuk:
#
   
nAngle=300
angle=np.zeros(nAngle)
Flong0=np.zeros(nAngle)
FlongB=np.zeros(nAngle)
Ftran0=np.zeros(nAngle)
FtranB=np.zeros(nAngle)

for i in range(nAngle):
   angle[i]=pi/2/nAngle*i
   Flong0[i]=2*math.cos(angle[i])
   Ftran0[i]=2*math.sin(angle[i])
   FlongB[i]=3*math.sin(angle[i])**2*math.cos(angle[i])
   FtranB[i]=(math.sin(angle[i])**2-2*math.cos(angle[i])**2)*math.sin(angle[i])
   

fig500=plt.figure(500)
line1, = plt.plot(angle,Flong0,'-r',linewidth=2,label='$F_{\||}$ for B=0')
line2, = plt.plot(angle,FlongB,'--r',linewidth=2,label='$F_{\|\|}$ for B=$\infty$')
line3, = plt.plot(angle,Ftran0,'-b',linewidth=2,label='$F_{\perp}$ for B=0')
line4, = plt.plot(angle,FtranB,'--b',linewidth=2,label='$F_{\perp}$ for B=$\infty$')
plt.title('Friction Force for Proton', color='m',fontsize=20)
plt.xlabel('Angle Relative Beam Axis, rad',color='m',fontsize=16)
plt.ylabel('Value Proporsed to Friction Force, rel.units',color='m',fontsize=16)
plt.legend(handler_map={line1: HandlerLine2D()},loc=4)
plt.grid(True)

#
# Verifying Parchomchuk's formulae from 
# Bruhwiler et al "Direct simulation ..." (Fig.3):
#
   
n_e=2.0e9                           # cm^-3
Bmag=5.0e4                          # Gs
q_e=4.8e-10                         # CGSE
m_e=9.1e-28                         # g
m_i=67.1e-24                        # g
c_light=2.99e10                     # cm/s
Ve_tranRMS=8.e8                     # cm/s
Ve_longRMS=1.e7                     # cm/s
tau=0.935e-9                        # s
w_p=np.sqrt(4.*pi*q_e**2*n_e/m_e)   # rad/s
tau_1=1/tau                         # 1/s
Ce=q_e**2/m_e                       # (cm/s)^2
w_L=q_e*Bmag/(c_light*m_e)          # rad/s
ro_L=Ve_tranRMS/w_L                 # cm
# Ve_eff=np.sqrt(Ve_tranRMS**2+Ve_longRMS**2)
Ve_eff=Ve_longRMS

print 'w_L/1e9, w_p/1e9, tau_1/1e9 (s^-1)=',w_L/1e9,w_p/1e9,tau_1/1e9
print 'Ce (cm^2/^2)=',Ce
print 'ro_L (cm)',ro_L


v_ion_tran=6.e6                          # cm/s
nVion=1000
v_ion_long=np.zeros(nVion)
v_ion=np.zeros(nVion)
dvParkh_long=np.zeros(nVion)
dvParkh_tran=np.zeros(nVion)
Vion_longMin=5.e5                         # cm/s
Vion_longMax=1.e8                         # cm/s

for i in range(nVion):
   v_ion_long[i]=Vion_longMin+(Vion_longMax-Vion_longMin)/nVion*i
   v_ion[i]=np.sqrt(v_ion_tran**2+v_ion_long[i]**2)
   ro_L=np.sqrt(Ve_tranRMS**2+Ve_longRMS**2)/w_L                 # cm
   dvParkh_long[i]=w_p**2*tau*q_e**2/m_i/pi*v_ion_long[i]/np.sqrt((v_ion[i]**2+Ve_eff**2)**3)* \
                   math.log((v_ion[i]/w_p+Ce/v_ion[i]**2+ro_L)/(Ce/v_ion[i]**2+ro_L))
   dvParkh_tran[i]=w_p**2*tau*q_e**2/m_i/pi*v_ion_tran/np.sqrt((v_ion[i]**2+Ve_eff**2)**3)* \
                   math.log((v_ion[i]/w_p+Ce/v_ion[i]**2+ro_L)/(Ce/v_ion[i]**2+ro_L))

fig600=plt.figure(600)
line1, = plt.loglog(v_ion_long,dvParkh_long,'-r',linewidth=2,label='$dV_{\||}$')
line2, = plt.loglog(v_ion_long,dvParkh_tran,'-b',linewidth=2,label='$dV_{\perp}$')
plt.title('Friction Force for Proton', color='m',fontsize=20)
plt.xlabel('$V_{ion\||}$, cm/s',color='m',fontsize=12)
plt.ylabel('$dV_{Parkh}$ cm/s',color='m',fontsize=16)
plt.legend(handler_map={line1: HandlerLine2D()},loc='lower center')
plt.grid(True)


ni=1000
x=np.zeros(ni)
y1=np.zeros(ni)
y2=np.zeros(ni)
for i in range(ni):
   x[i]=1./ni*i
   y1[i]=(1-x[i]**2)*x[i]
   y2[i]=(1-2*x[i]**2)*x[i]
   
fig700=plt.figure(700)
plt.plot(x,y1,'-r',x,y2,'-b')   
   
   
   
   





plt.show()   

sys.exit()   
   
   
   
   



