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
#       \vec deltaP_e(\vec {U}) - momentum ransfer from ion to electron due to collision,
#       f(\vec {V_e}) - electron velocity distribution function,
#       \rho - collision impact parameter;
#
# Value of \vec {deltaP_e(\vec U)} is calculated using "Magnus expand approach" (MEA)
#
#--------------------------------------------------------------------------------------

eVtoErg=1.602e-12                # energy from eV to erg (from CI to CGS)
# Indices:
(Ix, Ipx, Iy, Ipy, Iz, Ipz) = range(6)

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
kinEnergy_eBeam=470.*eVtoErg     # erg

stepsNumberOnGyro = 40           # number of the steps on each Larmour period

#
# Larmor frequency electron:
#
def omega_Larmor(mass,B_mag):
    return (q_elec)*B_mag/(mass*clight*1.e+2)             # rad/sec

#
# Derived quantities:
#
shiftV_e=np.sqrt(2.*kinEnergy_eBeam/m_elec)               # cm/sec
#
# The longitudinal average velocities of the electrons and ions are the same:
# 
kinEnergy_pBeam=kinEnergy_eBeam/m_elec*M_ion              # erg
shiftV_p=np.sqrt(2.*kinEnergy_pBeam/M_ion)                # cm/sec
print 'shiftV_e = %e, shiftV_p = %e' % (shiftV_e,shiftV_p)

tempRatio=Temp_eLong/Temp_eTran                           # dimensionless
velRatio=np.sqrt(tempRatio)                               # dimensionless
print 'tempRatio = %e, velRatio = %e' % (tempRatio,velRatio)


Omega_e = omega_Larmor(m_elec, B_mag)                     # rad/sec 
T_larm = 2*pi/Omega_e                                     # sec
timeStep = T_larm/stepsNumberOnGyro                       # time step, sec
print 'omega_Larmor= %e rad/sec, T_larm = %e sec, timeStep = %e sec' % (Omega_e,T_larm,timeStep)

rmsV_eTran = np.sqrt(2.*Temp_eTran*eVtoErg/m_elec)        # cm/sec
rmsV_eLong = np.sqrt(2.*Temp_eLong*eVtoErg/m_elec)        # cm/sec
print 'rmsV_eTran = %e cm/sec, rmsV_eLong = %e cm/sec' % (rmsV_eTran,rmsV_eLong)

ro_larm = rmsV_eTran/Omega_e                              # cm
print 'ro_larm = %e cm' % ro_larm

omega_e=np.sqrt(4*pi*n_eBeam*q_elec**2/m_elec)            # rad/sec
print 'omega_e = %e rad/sec' % omega_e

z_elec=np.zeros((6,numb_e))          # 1D vector: x_e,px_e,y_e,py_e,z_e,pz_e for each electron
z_ion=np.zeros((6,numb_p))           # 1D vector: x_i,px_i,y_i,py_i,z_i,pz_i for each proton

#
# Initial uniform distribution of the electron's impact parameter:
#
impctPar = np.random.uniform(high=a_eBeam,size=numb_e)    # cm

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
   z_elec[Ix,i]=impctPar[i]*math.cos(phi[i])              # cm
   z_elec[Iy,i]=impctPar[i]*math.sin(phi[i])              # cm

# Verifying of distribution:
plt.figure(20)
plt.plot(z_elec[Ix,:],z_elec[Iy,:],'.r',linewidth=2)
plt.xlabel('$x_e$, cm',color='m',fontsize=16)
plt.ylabel('$y_e$, cm',color='m',fontsize=16)
plt.title('Electron''s Initial Distribution',color='m',fontsize=16)
plt.xlim([np.min(z_elec[Ix,:]),np.max(z_elec[Ix,:])])
plt.ylim([np.min(z_elec[Iy,:]),np.max(z_elec[Iy,:])])
plt.grid(True)
plt.axes().set_aspect('equal')

#
# Initial gaussian distributions of the relative transverse electron's velocities 
#
z_elec[Ipx,:]=np.random.normal(scale=1.0,size=numb_e)             # vx_e/rmsV_eTran
z_elec[Ipy,:]=np.random.normal(scale=1.0,size=numb_e)             # vy_e/rmsV_eTran

# Verifying of distributions:
stdVex=z_elec[Ipx,:].std()
stdVey=z_elec[Ipy,:].std()
print 'stdVex = %e (must be 1.0), stdVey = %e (must be 1.0)' % (stdVex,stdVey)

plt.figure(30)
vel_hist=plt.hist(z_elec[Ipx,:],bins=30)
plt.xlabel('$V_{ex} / V_{e\perp}$',color='m',fontsize=16)
plt.ylabel('Particles',color='m',fontsize=16)
plt.ylim([0,1.1*np.max(vel_hist[0])])
plt.title(('Initial Distribution of $V_{ex} / V_{e\perp}$: $V_{rms}$ = %6.4f' % 1.), \
          color='m',fontsize=16)
plt.text(0.,1.025*np.max(vel_hist[0]),('From Distribution: $V_{rms}$ = %6.4f' % stdVex), \
         color='m',fontsize=16,ha='center')	  
plt.grid(True)

plt.figure(40)
vel_hist=plt.hist(z_elec[Ipy,:],bins=30)
plt.xlabel('$V_{ey} / V_{e\perp}$',color='m',fontsize=16)
plt.ylabel('Particles',color='m',fontsize=16)
plt.ylim([0,1.1*np.max(vel_hist[0])])
plt.title(('Initial Distribution of $V_{ey} / V_{e\perp}$: $V_{rms}$ = %6.4f' % 1.), \
          color='m',fontsize=16)
plt.text(0.,1.025*np.max(vel_hist[0]),('From Distribution: $V_{rms}$ = %6.4f' % stdVey), \
         color='m',fontsize=16,ha='center')	  
plt.grid(True)

#
# Initial gaussian distribution of the relative longitudinal electron's velocities 
#
relShiftV_e=shiftV_e/rmsV_eTran                                                 # dimensionless
z_elec[5,:]=np.random.normal(loc=relShiftV_e,scale=1.0*velRatio,size=numb_e)    # vz_e/rmsV_eTran

# Verifying of distribution:
avrVez=z_elec[Ipz,:].mean()
stdVez=z_elec[Ipz,:].std()
print 'avrVez = %e (must be %e),stdVexz = %e (must be %e)' % (avrVez,relShiftV_e,stdVez,velRatio)

plt.figure(50)
vel_hist=plt.hist(z_elec[Ipz,:],bins=30)
plt.xlabel('$V_{ez} / V_{e\perp}$',color='m',fontsize=16)
plt.ylabel('Particles',color='m',fontsize=16)
plt.ylim([0,1.1*np.max(vel_hist[0])])
plt.title(('Initial Distribution of $V_{ez} / V_{e\perp}$: <V> = %6.4f, $V_{rms}$ = %6.4f' \
           % (relShiftV_e,velRatio)), color='m',fontsize=16)
plt.text(relShiftV_e,1.025*np.max(vel_hist[0]), \
         ('From Distribution: <V>=%6.4f, $V_{rms}$ = %6.4f' % (avrVez,stdVez)), \
         color='m',fontsize=16,ha='center')	  
plt.grid(True)

#
# Returning to momenta:
#
z_elec[Ipx,:]=m_elec*rmsV_eTran*z_elec[1,:]                       # g*cm/sec
z_elec[Ipy,:]=m_elec*rmsV_eTran*z_elec[3,:]                       # g*cm/sec
z_elec[Ipz,:]=m_elec*rmsV_eTran*z_elec[5,:]                       # g*cm/sec

#
# 1D arrays with numb_e entries:
#
timeFlight=L_intrxn/(z_elec[Ipz,:]/m_elec)     # time of flight for each electron, sec
turnsNumber=timeFlight/T_larm                  # number of Larmour turns for each electron
stepsNumber=timeFlight/timeStep                # number of steps for each electron

#
# Convertion from electron's "coordinates" to guiding-center coordinates:
# For each electron z_e=(x_e,px_e,y_e,py_e,z_e,pz_e) --> zgc_e=(phi,p_phi,y_gc,py_gc,z_e,pz_e);
# z_c and zgc_e are 2D arrays with dimension (6,n_elec) 
#
def toGuidingCenter(z_e):
    mOmega=m_elec*Omega_e                                                        # g/sec
    zgc_e=z_e.copy()                                    # 2D array with dimension (6,n_elec)
    zgc_e[Ix,:] = np.arctan2(z_e[Ipx,:]+mOmega*z_e[Iy,:],z_e[Ipy,:])             # radians
    zgc_e[Ipx,:]= (((z_e[Ipx,:]+mOmega*z_e[Iy,:])**2+z_e[Ipy,:]**2)/(2.*mOmega)) # g*cm**2/sec
    zgc_e[Iy,:] =-z_e[Ipx,:]/mOmega                                              # cm
    zgc_e[Ipy,:]= z_e[Ipy,:]+mOmega*z_e[Ix,:]                                    # g/sec
    return zgc_e

#
# Convertion from guiding-center coordinates to electron's "coordinates":
# For each electron zgc_e=(phi,p_phi,y_gc,py_gc,z_e,pz_e) --> z_e=(x_e,px_e,y_e,py_e,z_e,pz_e);
# zgc_c and z_e are 2D arrays with dimension (6,n_elec) 
#
def fromGuidingCenter(zgc_e):
    mOmega=m_elec*Omega_e                                                        # g/sec
    rho_larm=np.sqrt(2.*zgc_e[Ipx,:]/mOmega)                                     # cm
    z_e = zgc.copy()                                    # 2D array with dimension (6,n_elec)
    z_e[Ix,:] = zgc_e[Ipy,:]/mOmega-rho_larm*np.cos(zgc_e[Ix,:])                 # cm
    z_e[Ipx,:]=-mOmega*zgc_e[Iy,:]                                               # g*cm/sec
    z_e[Iy,:] = zgc_e[Iy,:]+rho_larm*np.sin(zgc_e[Ix,:])                         # cm
    z_e[Ipy,:]= mOmega*rho_larm*np.cos(zgc_e[Ix,:])                              # g*cm/sec
    return z_e

#
# Initial ion (proton) coordinates: 
# all ions are placed in the beginning of the coordinate system
#          and have longitudinal velocity shiftV_p
#
z_ion[Ipz,:]=M_ion*shiftV_p           # all ions have the same momenta, g*cm/sec

#
# Matrix to dragg particle with mass 'm_part' through the drift during time interval 'deltaT':
#
def driftMatrix(mpart,deltaT):
    driftMtrx=np.identity(6)
    for i in (Ix,Iy,Iz):
       drftMtrx[i, i + 1]=deltaT/m_part                   # sec/g
    return drftMtrx

#
# Matrix to dragg electron through the solenoid with field 'B_mag' during time interval 'deltaT':
#
def solenoid_eMatrix(B_mag,deltaT):
    Omega_e=omega_Larmor(m_elec,B_mag)                    # rad/sec 
    mOmega= m_elec*Omega_e                                # g/sec
    phi=Omega_e*deltaT                                    # phase, rad
    cosPhi=math.cos(phi)                                  
    sinPhi=math.sin(phi)
    cosPhi_1=2.*math.sin(phi/2.)**2                       # 1-cos(a)=2sin^2(a/2)
    slndMtrx=np.identity(6)
    slndMtrx[Iy, Iy ]= cosPhi
    slndMtrx[Ipy,Ipy]= cosPhi
    slndMtrx[Iy, Ipy]= sinPhi/mOmega                      # sec/g
    slndMtrx[Ipy,Iy ]=-mOmega*sinPhi                      # g/sec
    slndMtrx[Iz, Ipz]= deltaT/m_elec                      # sec/g
    slndMtrx[Ix, Ipx]= sinPhi/mOmega                      # sec/g
    slndMtrx[Ix, Iy ]= sinPhi                             # dimensionless
    slndMtrx[Ix, Ipy]= cosPhi_1/mOmega                    # sec/g 
    slndMtrx[Iy, Ipx]=-cosPhi_1/mOmega                    # sec/g
    slndMtrx[Ipy,Ipx]=-sinPhi                             # dimensionless
    return slndMtrx

#
# Dragg electron and ion through the "collision" during time interval 'deltaT':
# During collision momenta of the particles sre changed by value deltaV_i/deltaR^3,
# where deltaV_i=Vion_i-Velec_i (i=x,y,z) and deltaR is distance between particles
#
def pullCollision(deltaT,z_i,z_e):
    g=deltaT*Z_ion*q_elec**2                              # g*cm^3/sec
    dz=z_i-z_e
    denom=(dz[Ix,:]**2+dz[Iy,:]**2+dz[Iz,:]**2)**(3/2)    # cm^3
    zf_i=z_i.copy()
    zf_e=z_e.copy()
    for ip in (Ipx,Ipy,Ipz):
       zf_i[ip,:]=z_i[ip,:]-g*dz[ip-1]/denom             # g*cm/sec
       zf_e[ip,:]=z_e[ip,:]+g*dz[ip-1]/denom             # g*cm/sec
    return zf_i,zf_e

plt.show()   

#
# Dragging all electrons near each protons:
#
zInit_ion=np.zeros(6)
zInit_elec=np.zeros(6)
for np in range(1):        # range(numb_p) 
   zInit_ion[:]=z_ion[:,np]
   print 'Dimension of zInit_ion=',zInit_ion.shape
   for ne in range(1):     # range(numb_e)  
      zInit_elec[:]=z_elec[:,ne]
      tol=L_intrxn/(zInit_elec[Ipz]/m_elec)               # time of flight for electron, sec
      trnsNmbr=int(tol/T_larm)                        # number of Larmour turns for electron
      stpsNmbr=int(tol/timeStep)                      # number of time steps for dragging
      print 'timeStep=%e, tol=%e,trnsNmbr=%d, stpsNmbr=%d' % (timeStep,tol,trnsNmbr,stpsNmbr) 
   
sys.exit()   

