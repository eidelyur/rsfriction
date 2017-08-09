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


eVtoErg=1.602e-12                # energy from eV to erg (from CI to CGS)
#
# Initial parameters:
#
Z_ion = qe*2.997e+9              # charge of ion (proton), CGSE units of the charge
M_ion = mp*1.e+3                 # mass of ion (proton), g
q_elec = qe*2.997e+9             # charge of electron, CGSE units of the charge
m_elec = me*1.e+3                # mass of electron, g
tangAlpha=1.                     # to calculate length of interaraction
B_mag = 2000.                    # magnetic field, Gs
Temp_eTran = 0.5                 # transversal temperature of electrons, eV
Temp_eLong = 2.e-4               # longitudinal temperature of electrons, eV
numb_e = 1000                    # number of electrons
numb_p = 50                      # number of protons

a_eBeam = 0.1                    # cm
n_eBeam = 1.e+9                  # cm^-3

stepsNumberOnGyro = 40           # number of the steps on each Larmour period

#
# Larmor frequency electron:
#
def omega_Larmor(mass,B_mag):
    return (q_elec)*B_mag/(mass*clight*1.e+2)             # rad/sec

#
# Derived quantities:
#
#
# The longitudinal shift velocities of the electrons and ions are the same:
# 
tempRatio=Temp_eLong/Temp_eTran                           # dimensionless
velRatio=np.sqrt(tempRatio)                               # dimensionless
print 'tempRatio = %e, velRatio = %e' % (tempRatio,velRatio)


omega_L = omega_Larmor(m_elec, B_mag)                     # rad/sec 
T_larm = 2*pi/omega_L                                     # sec
timeStep = T_larm/stepsNumberOnGyro                       # time step, sec
print 'omega_Larmor= %e rad/sec, T_larm = %e sec, timeStep = %e sec' % (omega_L,T_larm,timeStep)

rmsV_eTran = np.sqrt(2.*Temp_eTran*eVtoErg/m_elec)        # cm/sec
rmsV_eLong = np.sqrt(2.*Temp_eLong*eVtoErg/m_elec)        # cm/sec
print 'rmsV_eTran = %e cm/sec, rmsV_eLong = %e cm/sec' % (rmsV_eTran,rmsV_eLong)

ro_larm = rmsV_eTran/omega_L                              # cm
print '<ro_larm> = %e cm' % ro_larm

omega_e=np.sqrt(4*pi*n_eBeam*q_elec**2/m_elec)            # rad/sec
print 'omega_e = %e rad/sec' % omega_e

#
# Electrons are magnetized for impact parameter >> rhoCrit:
#
rhoCrit=math.pow(q_elec**2/(m_elec*omega_L**2),1./3)      # cm
maxLogRho=math.log10(a_eBeam/rhoCrit)
minLogRho=-1.

print 'rhoCrit = %e cm, maxLogRho = %e' % (rhoCrit,maxLogRho)


pointsRo=50
minB=.1                    # Gs
maxB=4000.                 # Gs

crrntB=np.zeros(pointsRo)
roCrtCrrnt=np.zeros(pointsRo)

for i in range(pointsRo):
   crrntB[i]=minB+(maxB-minB)/pointsRo*(i+1)
   freqL=q_elec*crrntB[i]/(m_elec*clight*1.e+2)
   roCrtCrrnt[i]=math.pow(q_elec**2/m_elec/freqL**2,1./3.)
   
fig5=plt.figure(5)
plt.plot(crrntB,1e+4*roCrtCrrnt,'-r',linewidth=2)   
plt.xlabel('B, Gs',color='m',fontsize=16)
plt.ylabel('$ro_{crit}$, $\mu$m',color='m',fontsize=16)
plt.title('Area of of Magnetization: $ro$ >> $ro_{crit}=[Z_ie^2/(m\cdot\omega_L^2]^{1/3}$',color='m',fontsize=16)
# plt.xlim([minLogRho,maxLogRho])
plt.grid(True)
   
   
   
pointsLog10=50
log10Rho=np.zeros(pointsLog10)
rhoCrrnt=np.zeros(pointsLog10)
omega_z=np.zeros(pointsLog10)
Omega=np.zeros(pointsLog10)
omega_p=np.zeros(pointsLog10)
omega_m=np.zeros(pointsLog10)
relOmegas=np.zeros(pointsLog10)

for i in range(pointsLog10):
   log10Rho[i]=minLogRho+(maxLogRho-minLogRho)/(pointsLog10-1)*i
   rhoCrrnt[i]=rhoCrit*math.pow(10.,log10Rho[i])
   omega_z[i]=np.sqrt(q_elec**2/(m_elec*rhoCrrnt[i]**3))
   Omega[i]=np.sqrt(omega_L**2+4.*omega_z[i]**2)
   relOmegas[i]=omega_z[i]/Omega[i]
   omega_p[i]=.5*(omega_L+Omega[i])
   omega_m[i]=.5*(omega_L-Omega[i])
   
print 'rhoCrrnt =', rhoCrrnt

'''
fig10=plt.figure(10)
plt.semilogy(range(pointsLog10),rhoCrrnt/rhoCrit,'.r')   
plt.xlabel('Point',color=','m',fontsize=16)
plt.ylabel('$ro/ro_{crit}$',color='m',fontsize=16)
plt.title('$ro/ro_{crit}$',color='m',fontsize=16)
# plt.xlim([minLogRho,maxLogRho])
plt.grid(True)
'''   
   
'''   
fig20=plt.figure(20)
plt.loglog(rhoCrrnt/rhoCrit,omega_z,'.r')   
plt.xlim([rhoCrrnt[0]/rhoCrit,rhoCrrnt[pointsLog10-1]/rhoCrit])
plt.xlabel('$ro/ro_{crit}$',color='m',fontsize=16)
plt.ylabel('$\omega_z$, sec',color='m',fontsize=16)
plt.title('$\omega_z$',color='m',fontsize=16)
plt.grid(True)
'''
   
fig30=plt.figure(30)
plt.loglog(rhoCrrnt/rhoCrit,omega_z/omega_L,'-r',linewidth=2)   
plt.xlim([rhoCrrnt[0]/rhoCrit,rhoCrrnt[pointsLog10-1]/rhoCrit])
plt.xlabel('Impact Parameter: $ro/ro_{crit}$; $ro_{crit}=[Z_ie^2/(m\cdot\omega_L^2]^{1/3}$',color='m',fontsize=16)
plt.ylabel('$\omega_z/\omega_L$',color='m',fontsize=16)
plt.title('$\omega_z/\omega_L=(ro/ro_{crit})^{-3/2}$',color='m',fontsize=16)
plt.grid(True)
   
'''   
fig40=plt.figure(40)
plt.semilogx(rhoCrrnt/rhoCrit,Omega,'.r')   
plt.xlim([rhoCrrnt[0]/rhoCrit,rhoCrrnt[pointsLog10-1]/rhoCrit])
plt.xlabel('$ro/ro_{crit}$',color='m',fontsize=16)
plt.ylabel('$\Omega$, sec',color='m',fontsize=16)
plt.grid(True)
'''
   
fig50=plt.figure(50)
plt.semilogx(rhoCrrnt/rhoCrit,Omega/omega_L,'-r',linewidth=2)   
plt.xlim([rhoCrrnt[0]/rhoCrit,rhoCrrnt[pointsLog10-1]/rhoCrit])
plt.xlabel('Impact Parameter: $ro/ro_{crit}$; $ro_{crit}=[Z_ie^2/(m\cdot\omega_L^2]^{1/3}$',color='m',fontsize=16)
plt.ylabel('$\Omega/\omega_L$',color='m',fontsize=16)
plt.title('$\Omega=[\omega_L^2+4\cdot\omega_z^2]^{1/2}=\omega_L\cdot[1+4/(ro/ro_{crit})^3]^{1/2}$', \
          color='m',fontsize=16)
plt.grid(True)

fig55=plt.figure(55)
plt.semilogx(rhoCrrnt/rhoCrit,Omega/omega_L,'-r',linewidth=2)   
plt.xlim([1.,rhoCrrnt[pointsLog10-1]/rhoCrit])
plt.ylim([0.9,2.5])
plt.xlabel('Impact Parameter: $ro/ro_{crit}$; $ro_{crit}=[Z_ie^2/(m\cdot\omega_L^2]^{1/3}$',color='m',fontsize=16)
plt.ylabel('$\Omega/\omega_L$',color='m',fontsize=16)
plt.title('$\Omega=[\omega_L^2+4\cdot\omega_z^2]^{1/2}=\omega_L\cdot[1+4/(ro/ro_{crit})^3]^{1/2}$', \
          color='m',fontsize=16)
plt.grid(True)

'''   
fig60=plt.figure(60)
plt.loglog(rhoCrrnt/rhoCrit,relOmegas,'-r')   
plt.xlim([rhoCrrnt[0]/rhoCrit,rhoCrrnt[pointsLog10-1]/rhoCrit])
plt.xlabel('$ro/ro_{crit}$',color='m',fontsize=16)
plt.ylabel('$\omega_z/\Omega$',color='m',fontsize=16)
plt.title('$\omega_z/\Omega$',color='m',fontsize=16)
plt.grid(True)
'''
   
   
N_ppt=40
turns=10
pointsPhi=N_ppt*turns
fi=np.zeros(pointsPhi)

Omega_omega_L=[np.sqrt(5.),1./0.577,1.2,1.00001]

ro=np.zeros(4)
ampl=np.zeros(4)
omega_p_omega_L=np.zeros(4)
omega_m_omega_L=np.zeros(4)

r=np.zeros((pointsPhi,4))              # dimensionless in unit ro_L
x=np.zeros((pointsPhi,4))              # dimensionless in unit ro_L
y=np.zeros((pointsPhi,4))              # dimensionless in unit ro_L

for j in range(4):
   ro[j]=math.pow(Z_ion*q_elec**2/m_elec/omega_L**2/(Omega_omega_L[j]**2-1),1./3.)
   omega_p_omega_L[j]=.5*(1+Omega_omega_L[j])
   omega_m_omega_L[j]=.5*(1-Omega_omega_L[j])
   ampl[j]=1.-1./Omega_omega_L[j]**2

print 'ro, ro/ro_c: ',  (ro, ro/rhoCrit)

# ampl=4.*omega_z_omegaL**2

stepT=2*pi/omega_L

# print ' stepT = %e, Omega_omegaL=%e:, omega_p_omega_L=%e, omega_m_omega_L=%e' % \
#       (stepT, Omega_omega_L,omega_p_omega_L,omega_m_omega_L)


for i in range(N_ppt*turns):
   fi[i]=2*pi/N_ppt*i
# print 'fi[0:41]= ', fi[0:41]

for j in range(4):
   pnt=0
   for turn in range(turns):
      for i in range(N_ppt):
         x[pnt,j]=(omega_p_omega_L[j]*math.cos(omega_m_omega_L[j]*fi[pnt])- \
                 omega_m_omega_L[j]*math.cos(omega_p_omega_L[j]*fi[pnt])+ \
	         math.cos(omega_p_omega_L[j]*fi[pnt])-math.cos(omega_m_omega_L[j]*fi[pnt]))/Omega_omega_L[j]
         y[pnt,j]=(omega_p_omega_L[j]*math.sin(omega_m_omega_L[j]*fi[pnt])+ \
                 omega_m_omega_L[j]*math.sin(omega_p_omega_L[j]*fi[pnt])+ \
	         math.sin(omega_p_omega_L[j]*fi[pnt])-math.sin(omega_m_omega_L[j]*fi[pnt]))/Omega_omega_L[j]
         r[pnt,j]=np.sqrt(1+ampl[j]*math.sin(.5*Omega_omega_L[j]*fi[pnt])**2)
         pnt += 1
   
fig70=plt.figure(70)
# plt.plot(fi[0:1.5*N_ppt],r[0:1.5*N_ppt,0],'-r',fi[0:1.5*N_ppt],r[0:1.5*N_ppt,1],'-b', \
#          fi[0:1.5*N_ppt],r[0:1.5*N_ppt,2],'-m',fi[0:1.5*N_ppt],r[0:1.5*N_ppt,3],'-g',linewidth=2) 
plt.plot(fi[0:1.5*N_ppt],r[0:1.5*N_ppt,0],'-r',fi[0:1.5*N_ppt],r[0:1.5*N_ppt,1],'-b', \
         fi[0:1.5*N_ppt],r[0:1.5*N_ppt,2],'-m',linewidth=2) 
plt.xlabel('$\phi=\omega_L\cdot t$',color='m',fontsize=16)
plt.ylabel('$r(t)/ro_L$',color='m',fontsize=16)
plt.title('$r(t)/ro_L=[1+4\cdot \omega_z^2/\Omega^2]^{1/2}$,  $\Omega=[\omega_L^2+4\cdot\omega_z^2]^{1/2}$',color='m',fontsize=16)
plt.legend(['$\Omega/\omega_L=2.236$','$\Omega/\omega_L=1.733$','$\Omega/\omega_L=1.200$'],fontsize=16,loc='upper right')
plt.grid(True)

fig70=plt.figure(75)
plt.plot(fi[0:1.5*N_ppt],(r[0:1.5*N_ppt,3]-1.)*1e+5,'-r',linewidth=2) 
plt.xlabel('$\phi=\omega_L\cdot t$',color='m',fontsize=16)
plt.ylabel('$10^5\cdot[r(t)/ro_L-1)]$',color='m',fontsize=16)
plt.title('$r(t)/ro_L=[1+4\cdot \omega_z^2/\Omega^2]^{1/2}$,  $\Omega=[\omega_L^2+4\cdot\omega_z^2]^{1/2}$',color='m',fontsize=16)
plt.legend(['$\Omega/\omega_L=1.00001$'],fontsize=16,loc='upper right')
plt.grid(True)

fig80=plt.figure(80)
plt.plot(x[0*N_ppt:1*N_ppt+1,0],y[0*N_ppt:1*N_ppt+1,0],'-r',linewidth=2) 
plt.hold(True)  
plt.plot(x[1*N_ppt:2*N_ppt+1,0],y[1*N_ppt:2*N_ppt+1,0],'-b',linewidth=2) 
plt.plot(x[2*N_ppt:3*N_ppt+1,0],y[2*N_ppt:3*N_ppt+1,0],'-m',linewidth=2) 
plt.plot(x[3*N_ppt:4*N_ppt+1,0],y[3*N_ppt:4*N_ppt+1,0],'-g',linewidth=2) 
plt.plot(x[4*N_ppt:5*N_ppt+1,0],y[4*N_ppt:5*N_ppt+1,0],'-k',linewidth=2) 
plt.plot(x[5*N_ppt:6*N_ppt+1,0],y[5*N_ppt:6*N_ppt+1,0],'-xr',linewidth=2,markersize=10) 
plt.plot(x[6*N_ppt:7*N_ppt+1,0],y[6*N_ppt:7*N_ppt+1,0],'-xb',linewidth=2,markersize=10) 
plt.plot(x[7*N_ppt:8*N_ppt+1,0],y[7*N_ppt:8*N_ppt+1,0],'-xm',linewidth=2,markersize=10) 
plt.plot(x[8*N_ppt:9*N_ppt+1,0],y[8*N_ppt:9*N_ppt+1,0],'-xg',linewidth=2,markersize=10) 
plt.xlabel('$x/ro_L$',color='m',fontsize=16)
plt.ylabel('$y/ro_L$',color='m',fontsize=16)
plt.title('First 9 Turns: $\Omega/\omega_L=2.236$, $ro/ro_{crit}=4.9\cdot10^{-4}$',color='m',fontsize=16)
plt.grid(True)
plt.axes().set_aspect('equal')
   
fig90=plt.figure(90)
plt.plot(x[0*N_ppt:1*N_ppt+1,1],y[0*N_ppt:1*N_ppt+1,1],'-r',linewidth=2) 
plt.hold(True)  
plt.plot(x[1*N_ppt:2*N_ppt+1,1],y[1*N_ppt:2*N_ppt+1,1],'-b',linewidth=2) 
plt.plot(x[2*N_ppt:3*N_ppt+1,1],y[2*N_ppt:3*N_ppt+1,1],'-m',linewidth=2) 
plt.plot(x[3*N_ppt:4*N_ppt+1,1],y[3*N_ppt:4*N_ppt+1,1],'-g',linewidth=2) 
plt.plot(x[4*N_ppt:5*N_ppt+1,1],y[4*N_ppt:5*N_ppt+1,1],'-k',linewidth=2) 
plt.plot(x[5*N_ppt:6*N_ppt+1,1],y[5*N_ppt:6*N_ppt+1,1],'-xr',linewidth=2,markersize=10) 
plt.plot(x[6*N_ppt:7*N_ppt+1,1],y[6*N_ppt:7*N_ppt+1,1],'-xb',linewidth=2,markersize=10) 
plt.plot(x[7*N_ppt:8*N_ppt+1,1],y[7*N_ppt:8*N_ppt+1,1],'-xm',linewidth=2,markersize=10) 
plt.plot(x[8*N_ppt:9*N_ppt+1,1],y[8*N_ppt:9*N_ppt+1,1],'-xg',linewidth=2,markersize=10) 
plt.xlabel('$x/ro_L$',color='m',fontsize=16)
plt.ylabel('$y/ro_L$',color='m',fontsize=16)
plt.title('First 9 Turns: $\Omega/\omega_L=1.733$, $ro/ro_{crit}=6.2\cdot10^{-4}$',color='m',fontsize=16)
plt.grid(True)
plt.axes().set_aspect('equal')
 
     
fig100=plt.figure(100)
plt.plot(x[0*N_ppt:1*N_ppt+1,2],y[0*N_ppt:1*N_ppt+1,2],'-r',linewidth=2) 
plt.hold(True)  
plt.plot(x[1*N_ppt:2*N_ppt+1,2],y[1*N_ppt:2*N_ppt+1,2],'-b',linewidth=2) 
plt.plot(x[2*N_ppt:3*N_ppt+1,2],y[2*N_ppt:3*N_ppt+1,2],'-m',linewidth=2) 
plt.plot(x[3*N_ppt:4*N_ppt+1,2],y[3*N_ppt:4*N_ppt+1,2],'-g',linewidth=2) 
plt.plot(x[4*N_ppt:5*N_ppt+1,2],y[4*N_ppt:5*N_ppt+1,2],'-k',linewidth=2) 
plt.plot(x[5*N_ppt:6*N_ppt+1,2],y[5*N_ppt:6*N_ppt+1,2],'-xr',linewidth=2,markersize=10) 
plt.plot(x[6*N_ppt:7*N_ppt+1,2],y[6*N_ppt:7*N_ppt+1,2],'-xb',linewidth=2,markersize=10) 
plt.plot(x[7*N_ppt:8*N_ppt+1,2],y[7*N_ppt:8*N_ppt+1,2],'-xm',linewidth=2,markersize=10) 
plt.plot(x[8*N_ppt:9*N_ppt+1,2],y[8*N_ppt:9*N_ppt+1,2],'-xg',linewidth=2,markersize=10) 
plt.xlabel('$x/ro_L$',color='m',fontsize=16)
plt.ylabel('$y/ro_L$',color='m',fontsize=16)
plt.title('First 9 Turns: $\Omega/\omega_L=1.002$, $ro/ro_{crit}=1.03\cdot10^{-3}$',color='m',fontsize=16)
plt.grid(True)
plt.axes().set_aspect('equal')
   
fig110=plt.figure(110)
plt.plot(x[0*N_ppt:1*N_ppt+1,3],y[0*N_ppt:1*N_ppt+1,3],'-r',linewidth=2) 
plt.hold(True)  
plt.plot(x[1*N_ppt:2*N_ppt+1,3],y[1*N_ppt:2*N_ppt+1,3],'-b',linewidth=2) 
plt.plot(x[2*N_ppt:3*N_ppt+1,3],y[2*N_ppt:3*N_ppt+1,3],'-m',linewidth=2) 
plt.plot(x[3*N_ppt:4*N_ppt+1,3],y[3*N_ppt:4*N_ppt+1,3],'-g',linewidth=2) 
plt.plot(x[4*N_ppt:5*N_ppt+1,3],y[4*N_ppt:5*N_ppt+1,3],'-k',linewidth=2) 
plt.plot(x[5*N_ppt:6*N_ppt+1,3],y[5*N_ppt:6*N_ppt+1,3],'-xr',linewidth=2,markersize=10) 
plt.plot(x[6*N_ppt:7*N_ppt+1,3],y[6*N_ppt:7*N_ppt+1,3],'-xb',linewidth=2,markersize=10) 
plt.plot(x[7*N_ppt:8*N_ppt+1,3],y[7*N_ppt:8*N_ppt+1,3],'-xm',linewidth=2,markersize=10) 
plt.plot(x[8*N_ppt:9*N_ppt+1,3],y[8*N_ppt:9*N_ppt+1,3],'-xg',linewidth=2,markersize=10) 
plt.xlabel('$x/ro_L$',color='m',fontsize=16)
plt.ylabel('$y/ro_L$',color='m',fontsize=16)
plt.title('First 9 Turns: $\Omega/\omega_L=1.00001$, $ro/ro_{crit}=2.9\cdot10^{-2}$',color='m',fontsize=16)
plt.grid(True)
plt.axes().set_aspect('equal')
   
fig120=plt.figure(120)
plt.plot(x[0*N_ppt:1*N_ppt+1,0],y[0*N_ppt:1*N_ppt+1,0],'-r',linewidth=2) 
plt.hold(True)  
plt.plot(x[0*N_ppt:1*N_ppt+1,1],y[0*N_ppt:1*N_ppt+1,1],'-b',linewidth=2) 
plt.plot(x[0*N_ppt:1*N_ppt+1,2],y[0*N_ppt:1*N_ppt+1,2],'-m',linewidth=2) 
plt.plot(x[0*N_ppt:1*N_ppt+1,3],y[0*N_ppt:1*N_ppt+1,3],'-g',linewidth=2) 
plt.xlabel('$x/ro_L$',color='m',fontsize=16)
plt.ylabel('$y/ro_L$',color='m',fontsize=16)
plt.title('First Turn for Different $ro/ro_{crit}$ with $ro_{crit}=[Z_ie^2/(m\omega_L^2)]^{1/3}$', \
          color='m',fontsize=16)
# plt.legend(['$ro/ro_{crit}=4.9\cdot10^{-4}$','$ro/ro_{crit}=6.2\cdot10^{-4}$', \
#             '$ro/ro_{crit}=1.03\cdot10^{-3}$','$ro/ro_{crit}=2.88\cdot10^{-2}$'],fontsize=16,loc='upper left')
plt.legend(['$4.9\cdot10^{-4}$','$6.2\cdot10^{-4}$','$1.0\cdot10^{-3}$','$2.9\cdot10^{-2}$'], \
           fontsize=16,loc='upper left')
plt.grid(True)
plt.axes().set_aspect('equal')
   
   
plt.show()   

sys.exit()   


