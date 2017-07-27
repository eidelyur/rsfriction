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
tangAlpha=1.                     # to calculate length of interaraction
B_mag = 2000.                    # magnetic field, Gs
Temp_eTran = 0.5                 # transversal temperature of electrons, eV
Temp_eLong = 2.e-4               # longitudinal temperature of electrons, eV
numb_e = 1000                    # number of electrons
numb_p = 50                      # number of protons

a_eBeam = 0.1                    # cm
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
# The longitudinal shift velocities of the electrons and ions are the same:
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
print '<ro_larm> = %e cm' % ro_larm

omega_e=np.sqrt(4*pi*n_eBeam*q_elec**2/m_elec)            # rad/sec
print 'omega_e = %e rad/sec' % omega_e

z_elec=np.zeros((6,numb_e)) # z_elec(6,:) is a vector: x_e,px_e,y_e,py_e,z_e,pz_e for each electron
z_ion=np.zeros((6,numb_p))  # z_ion(6,:)  is a vector: x_i,px_i,y_i,py_i,z_i,pz_i for each proton

#
# Initial uniform distribution of the electron's impact parameter (1D array):
#
impctPar = np.random.uniform(high=a_eBeam,size=numb_e)    # cm

avr_impctPar=1.e+4*impctPar.mean()

# Verifying of distribution:
plt.figure(10)
plt.hist(1.e+4*impctPar,bins=30)
plt.xlabel('Impact parameters, $\mu$m',color='m',fontsize=16)
plt.ylabel('Particles',color='m',fontsize=16)
plt.title(('Initial Impact Parameter (%d Particles): <>=%6.4f $\mu$m' % \
           (numb_e,avr_impctPar)),color='m',fontsize=16)
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
plt.title(('Electron''s Initial Distribution (%d Particles)' % numb_e),color='m',fontsize=16)
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
plt.title(('Initial Distribution of $V_{ex} / V_{e\perp}$ (%d Particles): $V_{rms}$ = %6.4f' \
           % (numb_e,1.)),color='m',fontsize=16)
plt.text(0.,1.025*np.max(vel_hist[0]),('From Distribution: $V_{rms}$ = %6.4f' % stdVex), \
         color='m',fontsize=16,ha='center')	  
plt.grid(True)

plt.figure(40)
vel_hist=plt.hist(z_elec[Ipy,:],bins=30)
plt.xlabel('$V_{ey} / V_{e\perp}$',color='m',fontsize=16)
plt.ylabel('Particles',color='m',fontsize=16)
plt.ylim([0,1.1*np.max(vel_hist[0])])
plt.title(('Initial Distribution of $V_{ey} / V_{e\perp}$ (%d Particles): $V_{rms}$ = %6.4f' \
           % (numb_e,1.)),color='m',fontsize=16)
plt.text(0.,1.025*np.max(vel_hist[0]),('From Distribution: $V_{rms}$ = %6.4f' % stdVey), \
         color='m',fontsize=16,ha='center')	  
plt.grid(True)

#
# Initial gaussian distribution of the relative longitudinal electron's velocities 
#
relShiftV_e=shiftV_e/rmsV_eLong                                                 # dimensionless
z_elec[Ipz,:]=np.random.normal(loc=relShiftV_e,scale=1.,size=numb_e)    # vz_e/rmsV_eLong

# Verifying of distribution:
avrVez=z_elec[Ipz,:].mean()
stdVez=z_elec[Ipz,:].std()
print 'avrVez = %e (must be %e),stdVez = %e (must be %e)' % (avrVez,relShiftV_e,stdVez,1.)

plt.figure(50)
vel_hist=plt.hist(z_elec[Ipz,:]-relShiftV_e,bins=30)
plt.xlabel('$V_{ez} / V_{e||}$',color='m',fontsize=16)
plt.ylabel('Particles',color='m',fontsize=16)
plt.ylim([0,1.1*np.max(vel_hist[0])])
plt.title(('Initial Distribution of $V_{ez} / V_{e||}$ (%d Particles): $V_{rms}$ = %6.4f'  % \
           (numb_e,1.)),color='m',fontsize=16)
plt.text(0,1.025*np.max(vel_hist[0]),('From Distribution: $V_{rms}$ = %6.4f' % stdVez), \
	 color='m',fontsize=16,ha='center')	  
plt.grid(True)

#
# 1D arrays with numb_e entries:
#
rhoLarm=np.zeros(numb_e)
L_intrcn=np.zeros(numb_e)
tol=np.zeros(numb_e)
trnsNmbr=np.zeros(numb_e)
stpsNmbr=np.zeros(numb_e)
for ne in range(numb_e):
   rhoLarm[ne]=1.e+4*rmsV_eTran*np.sqrt(z_elec[Ipx,ne]**2+z_elec[Ipy,ne]**2)/Omega_e     # mkm
   L_intrcn[ne]=2.*impctPar[ne]*tangAlpha                               # length of interaraction, cm
   tol[ne]=L_intrcn[ne]/np.abs((z_elec[Ipz,ne]*rmsV_eLong-shiftV_e))    # time of flight for, sec
   trnsNmbr[ne]=int(tol[ne]/T_larm)                                # number of Larmour turns
   stpsNmbr[ne]=int(tol[ne]/timeStep)                              # total number of steps
# for ne in range(numb_e):
#    print 'trnsNmbr: %d' % trnsNmbr[ne]

rms_rhoLarm=rhoLarm.std()
avr_Lintrcn=1.e+4*L_intrcn.mean()
minTrns=np.min(trnsNmbr)
maxTrns=np.max(trnsNmbr)
avrTrns=trnsNmbr.mean()
rmsTrns=trnsNmbr.std()
print 'minTrnsNmbr=%d, maxTrnsNmbr=%d, avrTrns=%d, rmsTrns=%d)' % (minTrns,maxTrns,avrTrns,rmsTrns)

plt.figure(60)
rhoL_hist=plt.hist(rhoLarm,bins=30)
plt.xlabel('$R_L$, $\mu$m',color='m',fontsize=16)
plt.ylabel('Particles',color='m',fontsize=16)
plt.ylim([0,1.025*np.max(rhoL_hist[0])])
plt.title(('Larmour Radius $R_L$ (%d Particles): rms = %6.3f $\mu$m' % (numb_e,rms_rhoLarm)), \
          color='m',fontsize=16)
plt.grid(True)

plt.figure(70)
intrcnL_hist=plt.hist(1.e+4*L_intrcn,bins=30)
plt.xlabel('$L_{interection}$, $\mu$m',color='m',fontsize=16)
plt.ylabel('Particles',color='m',fontsize=16)
plt.ylim([0,1.025*np.max(intrcnL_hist[0])])
plt.title(('Length of Interactions $L_{interaction}$ (%d Particles): <> = %6.3f $\mu$m' % \
           (numb_e,avr_Lintrcn)),color='m',fontsize=16)
plt.grid(True)

#
# Histogramm for number of turns larger then 100 and less than upperFactor*avrTrns
#
upperFactor=2.
upperTurns=upperFactor*avrTrns
trnsNmbrSpec=np.zeros(numb_e)
indxTrns=np.zeros(numb_e)
indxSkppd=np.zeros(numb_e)
neSlctd=-1
neSkppd=-1
for ne in range(numb_e):
   if 100 < trnsNmbr[ne] < upperTurns:
      neSlctd +=1
      trnsNmbrSpec[neSlctd]=trnsNmbr[ne]
      indxTrns[neSlctd]=ne
   else:
      neSkppd +=1
      indxSkppd[neSkppd]=ne

maxTrnsSpec=np.max(trnsNmbrSpec[1:neSlctd])
minTrnsSpec=np.min(trnsNmbrSpec[1:neSlctd])
print 'neSlctd=%d, maxTrnsSpec=%d, minTrnsSpec=%d: ' % (neSlctd,maxTrnsSpec,minTrnsSpec)
# print 'trnsNmbrSpec: ' , trnsNmbrSpec[1:neSlctd]
# print 'indxTrns: ' , indxTrns[1:neSlctd]
# print 'neSkppd, indxSkppd: ' , (neSkppd,indxSkppd[1:neSkppd])

plt.figure(80)
trns_hist=plt.hist(trnsNmbrSpec[1:neSlctd],bins=50)
plt.xlabel('Turns',color='m',fontsize=16)
plt.ylabel('Particles',color='m',fontsize=16)
plt.ylim([0,1.1*np.max(trns_hist[0])])
plt.xlim([50,2.*avrTrns])
plt.title(('Larmour Turns During Interaction (%d Particles): <>=%d' % (neSlctd,avrTrns)), \
          color='m',fontsize=16)
plt.text(0,1.025*np.max(trns_hist[0]), \
         ('  Selected Particles: 100 < Turns < 2$\cdot$Turns$_{avr}$ = %d' % upperTurns), \
	 color='m',fontsize=16,ha='left')	  
plt.grid(True)

# print 'sum(trns_hist): ',np.sum(trns_hist[0])

#
# Returning to momenta:
#
z_elec[Ipx,:]=m_elec*rmsV_eTran*z_elec[Ipx,:]                       # g*cm/sec
z_elec[Ipy,:]=m_elec*rmsV_eTran*z_elec[Ipy,:]                       # g*cm/sec
z_elec[Ipz,:]=m_elec*rmsV_eLong*(z_elec[Ipz,:]-relShiftV_e)         # g*cm/sec

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
def driftMatrix(m_part,deltaT,driftMtrx=[]):
    import numpy as np   # Next np.identity does not work without that! Why?!
    drftMtrx=np.identity(6)
    for i in (Ix,Iy,Iz):
       drftMtrx[i, i + 1]=deltaT/m_part                   # sec/g
    return drftMtrx

#
# Matrix to dragg electron through the solenoid with field 'B_mag' during time interval 'deltaT':
#
def solenoid_eMatrix(B_mag,deltaT):
    import numpy as np   # Next np.identity does not work without that! Why?!
    slndMtrx=np.identity(6)
    Omega_e=omega_Larmor(m_elec,B_mag)                    # rad/sec 
    mOmega= m_elec*Omega_e                                # g/sec
    phi=Omega_e*deltaT                                    # phase, rad
    cosPhi=math.cos(phi)                                  # dimensionless                                  
    sinPhi=math.sin(phi)                                  # dimensionless
    cosPhi_1=2.*math.sin(phi/2.)**2                       # dimensionless                      
    slndMtrx[Iy, Iy ]= cosPhi                             # dimensionless
    slndMtrx[Ipy,Ipy]= cosPhi                             # dimensionless
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
# During collision momenta of the particles sre changed by value deltaV_int/deltaR^3,
# where deltaV_int=Vion_i-Velec_i (i=x,y,z) and deltaR is a distance between particles
#
def draggCollision(deltaT,z_i,z_e):
    g=deltaT*Z_ion*q_elec**2                              # g*cm^3/sec
    dz=z_i-z_e
    denom=(dz[Ix]**2+dz[Iy]**2+dz[Iz]**2)**(3/2)    # cm^3
    zf_i=z_i.copy()
    zf_e=z_e.copy()
    for ip in (Ipx,Ipy,Ipz):
       zf_i[ip]=z_i[ip]-g*dz[ip-1]/denom             # g*cm/sec
       zf_e[ip]=z_e[ip]+g*dz[ip-1]/denom             # g*cm/sec
    return zf_i,zf_e

# plt.show()   

#
# Dragging all selected electrons near each protons:
#

zInit_ion=np.zeros(6)
zInit_elec=np.zeros(6)
#
# To draw trajectory of electron:
#
zFin_ion=np.zeros((6,201))
zFin_elec=np.zeros((6,201))
for np in range(1):                                    # range(numb_p) 
   zInit_ion[:]=z_ion[:,np]
   print 'Dimension of zInit_ion=',zInit_ion.shape
   for ne in range(1):                                 # range(neSlctd)
      neCrnt=indxTrns[ne]  
      zInit_elec[:]=z_elec[:,neCrnt]
#       rhoCrnt=rhoLarm[neCrnt]                                 # mkm
#       lenghtInt=L_intrcn[neCrnt]                              # length of interaraction, cm
      tolCrnt=tol[neCrnt]                                     # time of flight for, sec
      turnsCrnt=trnsNmbr[neCrnt]                              # number of Larmour turns
      stepsCrnt=stpsNmbr[neCrnt]                              # total number of steps
      print 'For %d: T_larm=%e, timeStep=%e, tol=%e,trnsNmbr=%d, stpsNmbr=%d' % \
            (neCrnt,T_larm,timeStep,tolCrnt,turnsCrnt,stepsCrnt) 
      z_ion=zInit_ion.copy()
      z_elec=zInit_elec.copy()
      zFin_ion[:,0]=z_ion.copy()
      zFin_elec[:,0]=z_elec.copy()
      matr_ion=driftMatrix(M_ion,.5*timeStep)
      matr_elec=solenoid_eMatrix(B_mag,.5*timeStep)
      for step in range(200):                           # range(stepsCrnt)
         z_ion=matr_ion.dot(z_ion)
	 z_elec=matr_elec.dot(z_elec)
         z_ion,z_elec=draggCollision(timeStep,z_ion,z_elec)
         z_ion=matr_ion.dot(z_ion)
	 z_elec=matr_elec.dot(z_elec)
         zFin_ion[:,step+1]=z_ion.copy()
         zFin_elec[:,step+1]=z_elec.copy()

print 'Dimension of zFin_elec, zFin_ion=',(zFin_elec.shape,zFin_ion.shape)

fig100=plt.figure(100)
ax100=fig100.gca(projection='3d')
ax100.plot(1.e+4*zFin_elec[Ix,:],1.e+4*zFin_elec[Iy,:],1.e+4*zFin_elec[Iz,:],'-r',linewidth=3)
plt.hold(True)
plt.title(('Electron: Impact Parameter=%5.3f cm, Larmour=%5.3f $\mu$m' \
           % (rhoLarm[indxTrns[0]],impctPar[indxTrns[0]])), color='m',fontsize=16)
plt.xlabel('x, $\mu m$',color='m',fontsize=16)
plt.ylabel('y, $\mu m$',color='m',fontsize=16)
# plt.zlabel('z, $\mu m$',color='m',fontsize=16)
# ax100.set_zlabel('z, $\mu m$',color='m',fontsize=20)
# ax100.text(-2.5,100,-442.5,'Larmor Circles',color='r',fontsize=16)
# ax100.text(-3.35,100,-483,'Larmor Center',color='b',fontsize=16)
# ax100.zaxis.label.set_color('magenta')
# ax100.zaxis.label.set_fontsize(16)
   
plt.show()   

sys.exit()   

