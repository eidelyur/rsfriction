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

#
# 2D marices for all electrons and ions:
#
z_elec=np.zeros((6,numb_e)) # z_elec(6,:) is a vector: x_e,px_e,y_e,py_e,z_e,pz_e for each electron
z_ion=np.zeros((6,numb_p))  # z_ion(6,:)  is a vector: x_i,px_i,y_i,py_i,z_i,pz_i for each proton

#
# Initial uniform distribution of the electron's impact parameter (1D array):
#
impctPar = np.random.uniform(high=a_eBeam,size=numb_e)    # cm
# print 'impctPar: ',impctPar

avr_impctPar=1.e+4*impctPar.mean()                        # mkm
print 'avr_impctPar: ',avr_impctPar

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
# Initial position of electrons: x=impactParameter, y=0:
#
for i in range(numb_e):
   z_elec[Ix,i]=impctPar[i]                                 # cm
   z_elec[Iy,i]=0.                                          # cm
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
relShiftV_e=shiftV_e/rmsV_eLong                                         # dimensionless
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
plt.title(('Shifted Initial Distribution of $V_{ez} / V_{e||}$ (%d Particles): $V_{rms}$ = %6.4f'  % \
           (numb_e,1.)),color='m',fontsize=16)
plt.text(0,1.025*np.max(vel_hist[0]),('$V_{e,shifted}/V_{e||}$ =%6.4f; From Distribution: $V_{rms}$ = %6.4f' % (relShiftV_e,stdVez)), \
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
   trnsNmbr[ne]=int(tol[ne]/T_larm)                                     # number of Larmour turns
   stpsNmbr[ne]=int(tol[ne]/timeStep)                                   # total number of steps
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
plt.ylim([0,1.1*np.max(intrcnL_hist[0])])
plt.title(('Interactions Length, $L_{interaction}$ (%d Particles): tan$_{interaction}$ = %5.3f' % \
           (numb_e,tangAlpha)), color='m',fontsize=16)
plt.text(avr_Lintrcn,1.025*np.max(intrcnL_hist[0]),('$<L_{interaction}>$ = %6.3f $\mu$m' % \
                     avr_Lintrcn), color='m',fontsize=16,ha='center')	  
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
      neSlctd=neSlctd+1
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
    denom=(dz[Ix]**2+dz[Iy]**2+dz[Iz]**2)**(3/2)          # cm^3
    zf_i=z_i.copy()
    zf_e=z_e.copy()
    for ip in (Ipx,Ipy,Ipz):
       zf_i[ip]=z_i[ip]-g*dz[ip-1]/denom                  # g*cm/sec
       zf_e[ip]=z_e[ip]+g*dz[ip-1]/denom                  # g*cm/sec
    return zf_i,zf_e

# plt.show()   

"""
#################################################################
#
#         Start of my debugging !!!
#
# To draw trajectory of electron:
#
#
# Returning to momenta:
#
z_elec[Ipx,:]=m_elec*rmsV_eTran*z_elec[Ipx,:]                       # g*cm/sec
z_elec[Ipy,:]=m_elec*rmsV_eTran*z_elec[Ipy,:]                       # g*cm/sec
z_elec[Ipz,:]=m_elec*rmsV_eLong*(z_elec[Ipz,:]-relShiftV_e)         # g*cm/sec
zInit_ion=np.zeros(6)
zInit_elec=np.zeros(6)
timeTrack=np.zeros(201)                                # to draw the figure(100)
zFin_ion=np.zeros((6,201))
zFin_elec=np.zeros((6,201))
for np in range(1):                                    # range(numb_p) 
   zInit_ion[:]=z_ion[:,np]
   for ne in range(1):                                 # range(neSlctd)
      neCrnt=indxTrns[ne]  
      zInit_elec[:]=z_elec[:,neCrnt]
      rhoCrnt=rhoLarm[neCrnt]                                 # mkm
      lenghtInt=L_intrcn[neCrnt]                              # length of interaraction, cm
      tolCrnt=tol[neCrnt]                                     # time of flight for, sec
      turnsCrnt=trnsNmbr[neCrnt]                              # number of Larmour turns
      stepsCrnt=stpsNmbr[neCrnt]                              # total number of steps
      vel=(float(zInit_elec[Ipx]**2)+float(zInit_elec[Ipy]**2))/m_elec**2
      print 'For %d: rho=%e, vel2=%e, T_larm=%e, timeStep=%e, tol=%e,trnsNmbr=%d, stpsNmbr=%d' % \
            (neCrnt,rhoCrnt,vel,T_larm,timeStep,tolCrnt,turnsCrnt,stepsCrnt) 
      z_ion_crnt=zInit_ion.copy()
      z_elec_crnt=zInit_elec.copy()
      zFin_ion[:,0]==zInit_ion.copy()
      zFin_elec[:,0]=zInit_elec.copy()
#       print 'ze=',zFin_elec[:,0]
      matr_ion=driftMatrix(M_ion,.5*timeStep)
      matr_elec=solenoid_eMatrix(B_mag,.5*timeStep)
      for step in range(200):                           # range(stepsCrnt)
         z_ion_crnt=matr_ion.dot(z_ion_crnt)
	 z_elec_crnt=matr_elec.dot(z_elec_crnt)
         z_ion_crnt,z_elec_crnt=draggCollision(timeStep,z_ion_crnt,z_elec_crnt)
         z_ion_crnt=matr_ion.dot(z_ion_crnt)
	 z_elec_crnt=matr_elec.dot(z_elec_crnt)
         zFin_ion[:,step+1]=z_ion_crnt.copy()
         zFin_elec[:,step+1]=z_elec_crnt.copy()
	 timeTrack[step+1]=timeTrack[step]+timeStep
# print 'Dimension of zFin_elec, zFin_ion=',(zFin_elec.shape,zFin_ion.shape)
# print 'ze_i=',zFin_elec[:,0]
# print 'ze_f=',zFin_elec[:,199]
# print 'xe(t)=',zFin_elec[Ix,:]
# print 'ye(t)=',zFin_elec[Iy,:]
zFin_elec_x=[]
zFin_elec_y=[]
zFin_elec_x=1.e+4*zFin_elec[Ix,:]
zFin_elec_y=1.e+4*zFin_elec[Iy,:]
zFin_elec_z=1.e+4*zFin_elec[Iz,:]
# print 't=',timeTrack
# print 'x=',zFin_elec_x
# print 'y=',zFin_elec_y
# print 'z=',zFin_elec_z
# dx=1.e+4*(np.max(zFin_elec_x)-np.min(zFin_elec_x))
# dy=1.e+4*(np.max(zFin_elec_y)-np.min(zFin_elec_y))
# print 'dx,dy: ',(dx,dy)
fig100=plt.figure(100)
ax100=fig100.gca(projection='3d')
ax100.plot(1.e+4*zFin_elec[Ix,:],1.e+4*zFin_elec[Iy,:],1.e+4*zFin_elec[Iz,:],'-r',linewidth=2)
plt.hold(True)
plt.title(('Electron: Impact Parameter=%5.3f cm, Larmour=%5.3f $\mu$m' \
           % (impctPar[indxTrns[0]],rhoLarm[indxTrns[0]])), color='m',fontsize=16)
plt.xlabel('x, $\mu m$',color='m',fontsize=16)
plt.ylabel('y, $\mu m$',color='m',fontsize=16)
# plt.zlabel('z, $\mu m$',color='m',fontsize=16)
# ax100.set_zlabel('z, $\mu m$',color='m',fontsize=20)
# ax100.text(-2.5,100,-442.5,'Larmor Circles',color='r',fontsize=16)
# ax100.text(-3.35,100,-483,'Larmor Center',color='b',fontsize=16)
# ax100.zaxis.label.set_color('magenta')
# ax100.zaxis.label.set_fontsize(16)
# fig110=plt.figure(110)
# plt.plot(timeElpsd,1.e+4*zFin_elec[Ix,:],'-r')
   
# fig120=plt.figure(120)
# plt.plot(timeElpsd,1.e+4*zFin_elec[Iy,:],'-r')
#
#         End of my debugging !!!
#
#################################################################
"""  

""" 
#################################################################
#
#         Start of my "dragging" game of electrons !!!
#
#
# 3D array "particles" - coordinate-vectors of all electrons for all time step 
# (to draw figure 90 only!):
#
# particles=array[[(xe,pxe,ye,pye,ze,pze)],[number of time steps],[number of electrons]]
#
particles=np.zeros((6,1000,numb_e))
cpuTime_p=np.zeros(numb_p)
cpuTime_e=np.zeros(numb_e)
elpsdTime_p=np.zeros(numb_p)
elpsdTime_e=np.zeros(numb_e)
#
# Returning to momenta:
#
z_elec[Ipx,:]=m_elec*rmsV_eTran*z_elec[Ipx,:]                       # g*cm/sec
z_elec[Ipy,:]=m_elec*rmsV_eTran*z_elec[Ipy,:]                       # g*cm/sec
z_elec[Ipz,:]=m_elec*rmsV_eLong*(z_elec[Ipz,:]-relShiftV_e)         # g*cm/sec
#
# Dragging all selected electrons near each protons:
#
deltaPion=np.zeros((3,numb_p))
matr_ion=driftMatrix(M_ion,.5*timeStep)
matr_elec=solenoid_eMatrix(B_mag,.5*timeStep)
for np in range(1):                                    # range(numb_p): 
   for ne in range(5):                                 # range(neSlctd):
      timeStart=os.times()
      neCrnt=indxTrns[ne] 
      stepsCrnt=int(stpsNmbr[neCrnt])                         # total number of steps
# For debugging:      
#       rhoCrnt=rhoLarm[neCrnt]                                 # mkm
#       lenghtInt=L_intrcn[neCrnt]                              # length of interaraction, cm
#       tolCrnt=tol[neCrnt]                                     # time of flight for, sec
#       turnsCrnt=int(trnsNmbr[neCrnt])                         # number of Larmour turns
#       vel=(float(z_elec[Ipx,neCrnt]**2)+float(z_elec[Ipy,neCrnt]**2))/m_elec**2
#       print 'For %d: rho=%e, vel2=%e, T_larm=%e, timeStep=%e, tol=%e,trnsNmbr=%d, stpsNmbr=%d' % \
#             (neCrnt,rhoCrnt,vel,T_larm,timeStep,tolCrnt,turnsCrnt,stepsCrnt) 
      z_ion_crnt=z_ion[:,np]
      z_elec_crnt=z_elec[:,neCrnt]
      zFin_ion=[z_ion]
      zFin_elec=[z_elec]
      zFin_ion.append(z_ion_crnt)
      zFin_elec.append(z_elec_crnt)
      for step in range(stepsCrnt):
         z_ion_crnt=matr_ion.dot(z_ion_crnt)
	 z_elec_crnt=matr_elec.dot(z_elec_crnt)
         z_ion_crnt,z_elec_crnt=draggCollision(timeStep,z_ion_crnt,z_elec_crnt)
         z_ion_crnt=matr_ion.dot(z_ion_crnt)
	 z_elec_crnt=matr_elec.dot(z_elec_crnt)
         deltaPion[0,np] +=z_ion_crnt[Ipx]-z_ion[Ipx,np]
         deltaPion[1,np] +=z_ion_crnt[Ipy]-z_ion[Ipy,np]
         deltaPion[2,np] +=z_ion_crnt[Ipz]-z_ion[Ipz,np]
         zFin_ion.append(z_ion_crnt)
         zFin_elec.append(z_elec_crnt)
         particles[:,step,ne]=zFin_elec[step+1]
      timeEnd=os.times()
      cpuTime_e[ne]   +=float(timeEnd[0])-float(timeStart[0])     # CPU time for electron
      elpsdTime_e[ne] +=float(timeEnd[4])-float(timeStart[4])     # elapsed real time for electron
      cpuTime_p[np]   +=cpuTime_e[ne]                             # CPU time for proton
      elpsdTime_p[np] +=elpsdTime_e[ne]                           # elapsed real time for proton
      print 'Electron %d: steps = %d, cpu(s) = %8.4f, elapsed(s) = %8.4f, cpu/step(mks) = %6.3f' % \
            (neCrnt,stpsNmbr[neCrnt],cpuTime_e[ne],elpsdTime_e[ne], \
	     1.e+6*cpuTime_e[ne]/(1.*stpsNmbr[neCrnt]))
      
   print '        Proton %d: electrons = %d, cpu(s) = %8.4f, elapsed(s) = %8.4f' % \
         (np,neSlctd,cpuTime_p[np],elpsdTime_p[np])
fig90=plt.figure(90)
ax90=fig90.gca(projection='3d')
ax90.plot(1.e+4*particles[Ix,0:200,0],1.e+4*particles[Iy,0:200,0],1.e+4*particles[Iz,0:200,0],'-r', \
          linewidth=2)
plt.hold(True)
# for k in range(4):
#    ax90.plot(1.e+4*particles[Ix,0:200,k+1],1.e+4*particles[Iy,0:200,k+1],1.e+4*particles[Iz,0:200,k+1],'-r', \
#              linewidth=2)
ax90.plot(1.e+4*particles[Ix,0:200,1],1.e+4*particles[Iy,0:200,1],1.e+4*particles[Iz,0:200,1],'-b', \
          linewidth=2)
ax90.plot(1.e+4*particles[Ix,0:200,2],1.e+4*particles[Iy,0:200,2],1.e+4*particles[Iz,0:200,2],'-m', \
          linewidth=2)
ax90.plot(1.e+4*particles[Ix,0:200,3],1.e+4*particles[Iy,0:200,3],1.e+4*particles[Iz,0:200,3],'-g', \
          linewidth=2)
ax90.plot(1.e+4*particles[Ix,0:200,4],1.e+4*particles[Iy,0:200,4],1.e+4*particles[Iz,0:200,4],'-k', \
          linewidth=2)
plt.title(('Electrons\nParticle 0: Impact Parameter=%5.3f $\mu$m, $R_{L}$=%5.3f $\mu$m \
                     \nParticle 1: Impact Parameter=%5.3f $\mu$m, $R_{L}$=%5.3f $\mu$m \
                     \nParticle 2: Impact Parameter=%5.3f $\mu$m, $R_{L}$=%5.3f $\mu$m \
                     \nParticle 3: Impact Parameter=%5.3f $\mu$m, $R_{L}$=%5.3f $\mu$m \
                     \nParticle 4: Impact Parameter=%5.3f $\mu$m, $R_{L}$=%5.3f $\mu$m' % \
           (1.e+4*impctPar[indxTrns[0]],rhoLarm[indxTrns[0]], \
	    1.e+4*impctPar[indxTrns[1]],rhoLarm[indxTrns[1]], \
	    1.e+4*impctPar[indxTrns[2]],rhoLarm[indxTrns[2]], \
	    1.e+4*impctPar[indxTrns[3]],rhoLarm[indxTrns[3]], \
	    1.e+4*impctPar[indxTrns[4]],rhoLarm[indxTrns[4]])), color='m',fontsize=6)
plt.xlabel('x, $\mu m$',color='m',fontsize=16)
plt.ylabel('y, $\mu m$',color='m',fontsize=16)
#
#         End of my "dragging" game !!!
#
#################################################################
"""

#================================================================
#
#  Integrations to calculate the friction force
#z_elec[Ipx,:]
#
binsImpPar=30
binsVe_tran=30
binsVe_long=30

impPar_slctd=np.zeros(binsImpPar)
impPar_step=a_eBeam/binsImpPar
for k in range(binsImpPar):
   impPar_slctd[k]=impPar_step*(k+.5)

#
# Initial gaussian distributions of the relative transverse electron's velocity 
# in polar velocities coordinates: 
#
ve_tran=np.zeros(numb_e)
for i in range(numb_e):
   ksi1=np.random.uniform(low=0.,high=1.,size=1)
   ksi2=np.random.uniform(low=0.,high=1.,size=1)
   z_elec[Ipx,i]=np.sqrt(-2.*math.log(ksi1))*math.cos(2.*pi*ksi2)  # dimensionless: eVx_tran/rmsV_eTran
   z_elec[Ipy,i]=np.sqrt(-2.*math.log(ksi1))*math.sin(2.*pi*ksi2)  # dimensionless: eVy_tran/rmsV_eTran
   ve_tran[i]=np.sqrt(z_elec[Ipx,i]**2+z_elec[Ipy,i]**2)          # dimensionless: ve_tran/rmsV_eTran

# print 'z_elec[Ipx,:]=',z_elec[Ipx,:]
# print 'z_elec[Ipy,:]=',z_elec[Ipy,:]
stdVtran=ve_tran.std()
print 'stdVtran=',stdVtran

minVe_tran=np.min(ve_tran)                                   # dimensionless
maxVe_tran=np.max(ve_tran)                                   # dimensionless

stepVe_tran=(maxVe_tran-minVe_tran)/binsVe_tran              # dimensionless

velEtran_slctd=np.zeros(binsVe_tran)                         # dimensionless
for k in range(binsVe_tran):
   velEtran_slctd[k]=minVe_tran+stepVe_tran*(k+.5)           # dimensionless

print 'minVe_tran, maxVe_tran = ', (minVe_tran,maxVe_tran)
# print 'velEtran_slctd = ',velEtran_slctd

plt.figure(110)
velTran_hist=plt.hist(ve_tran,bins=30)
plt.xlabel('$V_{\perp} / V_{e\perp}$',color='m',fontsize=16)
plt.ylabel('Particles',color='m',fontsize=16)
plt.ylim([0,1.025*np.max(velTran_hist[0])])
plt.title(('Initial Distribution of $V_{\perp} / V_{e\perp}$ (%d Particles)'  % numb_e), \
          color='m',fontsize=16)
# plt.text(0,1.025*np.max(velTran_hist[0]),('From Distribution: $V_{rms}$ = %6.4f' % stdVtran), \
# 	 color='m',fontsize=16,ha='center')	  
plt.grid(True)

#
# Initial uniform distribution of the transverse velocity in cross section:
#
phi=np.random.uniform(high=2*pi,size=numb_e)
for i in range(numb_e):
   fi=np.random.uniform(high=2*pi,size=1)
   z_elec[Ipx,i]=ve_tran[i]*math.cos(fi)                            # dimensionless 
   z_elec[Ipy,i]=ve_tran[i]*math.sin(fi)                            # dimensionless 

#
# Initial distribution of the longitudinal velocity:
#
minVe_long=np.min(z_elec[Ipz,:])                                    # dimensionless
maxVe_long=np.max(z_elec[Ipz,:])                                    # dimensionless
print 'minVe_long, maxVe_long = ',(minVe_long,maxVe_long)

velElong_slctd=np.zeros(binsVe_long)                                # dimensionless
stepVe_long=(maxVe_long-minVe_long)/binsVe_long                     # dimensionless

for k in range(binsVe_long):
   velElong_slctd[k]=minVe_long+stepVe_long*(k+.5)                  # dimensionless
print 'Relative velElong_slctd =\n',velElong_slctd
print 'Absolute velElong_slctd (cm/sec) =\n' , rmsV_eLong*velElong_slctd-shiftV_e

#
# Specific data for selected values for integration: 
# interaction length, time of flight, total number of time steps 
#
intrLen_slctd=np.zeros(binsImpPar)
tol_slctd=np.zeros((binsImpPar,binsVe_long))
tmStep_slctd=np.zeros((binsImpPar,binsVe_long))
for j in range(binsImpPar):
   intrLen_slctd[j]=2.*impPar_slctd[j]*tangAlpha                     # length of interaction, cm
   for k in range(binsVe_long):
      velLongCrnt=rmsV_eLong*velElong_slctd[k]                       # vLong, cm/sec
      tol_slctd[j,k]=intrLen_slctd[j]/np.abs(velLongCrnt-shiftV_e)   # time of flight for electron, sec
      tmStep_slctd[j,k]=int(tol_slctd[j,k]/timeStep)                 # total number of steps
print 'intrLen_slctd (cm):\n', intrLen_slctd
for j in range(binsImpPar):
   print 'Times of flight for length interaction %e cm' % intrLen_slctd[j]
   print 'Times are:\n', tol_slctd[j,:]
for j in range(binsImpPar):
   print 'Total number of time steps for length interaction %e cm' % intrLen_slctd[j]
   print 'Numbers are:\n', tmStep_slctd[j,:]
	    
#
# Returning to momenta:
#
z_elec[Ipx,:]=m_elec*rmsV_eTran*z_elec[Ipx,:]                       # g*cm/sec
z_elec[Ipy,:]=m_elec*rmsV_eTran*z_elec[Ipy,:]                       # g*cm/sec
z_elec[Ipz,:]=m_elec*rmsV_eLong*(z_elec[Ipz,:]-relShiftV_e)         # g*cm/sec

#
# Preparation for integration:
#
factorIntgrtn_init=np.sqrt(2.*pi)*impPar_step*stepVe_tran*stepVe_long   # cm*(cm/sec)**2
factorIntgrtn_init *= n_eBeam/(rmsV_eTran**2*rmsV_eLong)            # sec/cm**3
print 'factorIntgrtn_init (sec/cm^3)= ',factorIntgrtn_init

rhoLarm_slctd=np.zeros(binsVe_tran)
particles=np.zeros((6,10000,binsImpPar))                             # to draw the trajectories
cpuTime_p=np.zeros(numb_p)
cpuTime_e=np.zeros(binsImpPar)
elpsdTime_p=np.zeros(numb_p)
elpsdTime_e=np.zeros(binsImpPar)

deltaPion=np.zeros((3,numb_p))
dPion_crnt=np.zeros(3)
frctnForce=np.zeros((3,numb_p))
matr_ion=driftMatrix(M_ion,.5*timeStep)
print 'matr_ion: ',matr_ion
matr_elec=solenoid_eMatrix(B_mag,.5*timeStep)
print 'matr_elec: ',matr_elec

#
# Integration along the larmour trajectories:
#
gFactor=timeStep*Z_ion*q_elec**2                                    # g*cm^3/sec
z_elec_crnt=np.zeros(6)                                             # Initial vector for electron
z_ion_crnt=np.zeros(6)                                              # Initial vector for ion
for nion in range(1):                       # range(numb_p):
   for m in range(6):
      z_ion_crnt[m]=0.                                              # Initial zero-vector for ion
# All ions have the same longitudinal momenta:                        
#    z_ion_crnt[Ipz]=M_ion*shiftV_p                                   # pz, g*cm/sec                                                              
   for j in range(1):                    # range(binsImpPar):
      factorIntgrtn_crnt=impPar_slctd[j]                            # cm
      intrcnLength=intrLen_slctd[j]                                 # length of interaction, cm
      for k in range(1):                  # range(binsVe_long):
         velLongCrnt=rmsV_eLong*velElong_slctd[k]                   # vLong, cm/sec
         print 'relShiftV_e = %e, velElong_slctd[k] = %e, shiftV_p = %e, shiftV_e = %e, velLongCrnt = %e: ' % \
	       (relShiftV_e,velElong_slctd[k],shiftV_p,shiftV_e,velLongCrnt)
         timeOfLight=tol_slctd[j,k]                                 # time of flight for electron, sec
         timeSteps=int(tmStep_slctd[j,k])                           # total number of steps
         print 'tmStep_slctd[j,k], timeSteps: ',(tmStep_slctd[j,k],timeSteps)
         for i in range(5):                  # range(binsVe_tran):
	    timeStart=os.times()
            for m in range(6):
               z_elec_crnt[m]=0.                                          # Initial zero-vector for electron
#
# Initial position of electrons: x=impactParameter, y=0:
#
            z_elec_crnt[Ix]=impPar_slctd[j]                               # x, cm
            z_elec_crnt[Iz]=-.5*intrcnLength                              # z, cm
 	    numbCrntElec=binsVe_long*(binsVe_tran*j+k)+i
            rhoLarm_slctd[i]=rmsV_eTran*velEtran_slctd[i]/Omega_e          # cm          
            velTranCrnt=rmsV_eTran*velEtran_slctd[i]                       # vTran, cm/sec
	    factorIntgrtn_crnt *= velTranCrnt                              # cm*cm/sec
#            print 'factorIntgrtn_crnt %e ' % factorIntgrtn_crnt 
	    absDeltaV=np.sqrt(velTranCrnt**2+(shiftV_p-velLongCrnt)**2)    # |Vion-Velec|, cm/sec
	    factorIntgrtn_crnt *= absDeltaV                                # cm*(cm/sec)**2
# For checking of the trajectories:
#            z_elec_crnt[Ipx]=0.                                            # px, g*cm/sec
#            z_elec_crnt[Ipy]=m_elec*velTranCrnt                            # py, g*cm/sec
            phi=np.random.uniform(low=0.,high=2.*pi,size=1)
            z_elec_crnt[Ipx]=m_elec*velTranCrnt*math.cos(phi)              # px, g*cm/sec
            z_elec_crnt[Ipy]=m_elec*velTranCrnt*math.sin(phi)              # py, g*cm/sec
            z_elec_crnt[Ipz]=m_elec*(velLongCrnt-shiftV_e)                            # pz, g*cm/sec
            zFin_ion=[z_ion_crnt]
            zFin_elec=[z_elec_crnt]
            zFin_ion.append(z_ion_crnt)
            zFin_elec.append(z_elec_crnt)
#
# Dragging  electron near each protons:
#
#             print 'Electron %d: steps = %d, rhoLarm(mkm) = %8.6f' % \
#                   (numbCrntElec,timeSteps,1.e+4*rhoLarm_slctd[i])
            pointTrack=0
            for istep in range(timeSteps):
#
# Before interaction:
#	       
               z_ion_crnt=matr_ion.dot(z_ion_crnt)
 	       z_elec_crnt=matr_elec.dot(z_elec_crnt)
# To draw ion and first 4 electron trajectories for checking:
#                particles[:,pointTrack,numbCrntElec+1]=zFin_elec[istep+1]
               particles[:,pointTrack,numbCrntElec+1]=z_elec_crnt
               if numbCrntElec==0:	       
                  particles[:,pointTrack,0]=z_ion_crnt
               pointTrack += 1
#----------------	       
#
# Interaction between ion and electron:
#	       
###               z_ion_crnt,z_elec_crnt=draggCollision(timeStep,z_ion_crnt,z_elec_crnt)
               dz=z_ion_crnt-z_elec_crnt
               denom=(dz[Ix]**2+dz[Iy]**2+dz[Iz]**2)**(3/2)                # cm^3
               for ip in (Ipx,Ipy,Ipz):
                  dPion_crnt[ip//2] = -gFactor*dz[ip-1]/denom              # g*cm/sec
                  z_ion_crnt[ip] =z_ion_crnt[ip] +dPion_crnt[ip//2]        # g*cm/sec
                  z_elec_crnt[ip]=z_elec_crnt[ip]-dPion_crnt[ip//2]        # g*cm/sec
                  deltaPion[ip//2,nion] += dPion_crnt[ip//2]               # g*cm/sec
#
#----------------
# To draw ion and first 4 electron trajectories for checking:
#                particles[:,pointTrack,numbCrntElec+1]=zFin_elec[istep+1]
               particles[:,pointTrack,numbCrntElec+1]=z_elec_crnt
               if numbCrntElec==0:	       
                  particles[:,pointTrack,0]=z_ion_crnt
               pointTrack += 1
#
# After interaction:
#	       
               z_ion_crnt=matr_ion.dot(z_ion_crnt)
 	       z_elec_crnt=matr_elec.dot(z_elec_crnt)
# To draw ion and first 4 electron trajectories for checking:
#                particles[:,pointTrack,numbCrntElec+1]=zFin_elec[istep+1]
               particles[:,pointTrack,numbCrntElec+1]=z_elec_crnt
               if numbCrntElec==0:	       
                  particles[:,pointTrack,0]=z_ion_crnt
               pointTrack += 1
###               deltaPx_ion = z_ion_crnt[Ipx]-z_ion[Ipx,nion]               # deltaPx, g*cm/sec
###               deltaPy_ion = z_ion_crnt[Ipy]-z_ion[Ipy,nion]               # deltaPy, g*cm/sec
###               deltaPz_ion = z_ion_crnt[Ipz]-z_ion[Ipz,nion]               # deltaPz, g*cm/sec
#                print 'deltaPion (step %d): %e %e %e ' % (istep,deltaPx_ion,deltaPy_ion,deltaPz_ion)
               zFin_ion.append(z_ion_crnt)
               zFin_elec.append(z_elec_crnt)
# To draw ion and first 4 electron trajectories for checking:
#                particles[:,istep,numbCrntElec+1]=zFin_elec[istep+1]
#                if numbCrntElec==0:	       
#                   particles[:,istep,0]=zFin_ion[istep+1]
###	       frctnForce[0,nion] += factorIntgrtn_crnt*deltaPx_ion        # g*cm*(cm/sec)**3
###	       frctnForce[1,nion] += factorIntgrtn_crnt*deltaPy_ion        # g*cm*(cm/sec)**3
###	       frctnForce[2,nion] += factorIntgrtn_crnt*deltaPz_ion        # g*cm*(cm/sec)**3
#                print 'Friction force (step %d): %e %e %e ' % \
# 	             (istep,frctnForce[0,nion],frctnForce[1,nion],frctnForce[2,nion])
#            print 'Friction force: %e %e %e ' % (frctnForce[0,nion],frctnForce[1,nion],frctnForce[2,nion])
            timeEnd=os.times()
            cpuTime_e[numbCrntElec]   += float(timeEnd[0])-float(timeStart[0])   # CPU time for electron
            elpsdTime_e[numbCrntElec] += float(timeEnd[4])-float(timeStart[4])   # elapsed real time for electron
            cpuTime_p[nion]   += cpuTime_e[numbCrntElec]                         # CPU time for proton
            elpsdTime_p[nion] += elpsdTime_e[numbCrntElec]                       # elapsed real time for proton
            print 'Electron %d: steps = %d, cpu(s) = %e, elapsed(s) = %e, cpu/step(mks) = %e' % \
                  (numbCrntElec,timeSteps,cpuTime_e[numbCrntElec],elpsdTime_e[numbCrntElec], \
		   1.e+6*cpuTime_e[numbCrntElec]/timeSteps)
###   frctnForce[0,nion] = frctnForce[0,nion]*factorIntgrtn_init                    # g*cm/sec**2
###   frctnForce[1,nion] = frctnForce[1,nion]*factorIntgrtn_init                    # g*cm/sec**2
###   frctnForce[2,nion] = frctnForce[2,nion]*factorIntgrtn_init                    # g*cm/sec**2
   print '        Proton %d: electrons = %d, cpu(s) = %e, elapsed(s) = %e' % \
         (nion,numbCrntElec+1,cpuTime_p[nion],elpsdTime_p[nion])
   print 'deltaPion: (ion %d) %e %e %e ' % (nion,deltaPion[0,nion],deltaPion[1,nion],deltaPion[2,nion])

points=pointTrack
print 'points=%d' % points

fig120=plt.figure(120)
ax120=fig120.gca(projection='3d')
ax120.plot(1.e+4*particles[Ix,0:points,0],1.e+4*particles[Iy,0:points,0],1.e+4*particles[Iz,0:points,0],'ok', \
          linewidth=6)
plt.hold(True)
# for k in range(4):
#    ax120.plot(1.e+4*particles[Ix,0:200,k+1],1.e+4*particles[Iy,0:200,k+1],1.e+4*particles[Iz,0:200,k+1],'-r', \
#              linewidth=2)
ax120.plot(1.e+4*particles[Ix,0:points,1],1.e+4*particles[Iy,0:points,1],1.e+4*particles[Iz,0:points,1],'-r', \
          linewidth=2)
ax120.plot(1.e+4*particles[Ix,0:points,2],1.e+4*particles[Iy,0:points,2],1.e+4*particles[Iz,0:points,2],'-b', \
          linewidth=2)
ax120.plot(1.e+4*particles[Ix,0:points,3],1.e+4*particles[Iy,0:points,3],1.e+4*particles[Iz,0:points,3],'-m', \
          linewidth=2)
ax120.plot(1.e+4*particles[Ix,0:points,4],1.e+4*particles[Iy,0:points,4],1.e+4*particles[Iz,0:points,4],'-g', \
          linewidth=2)
plt.title(('Electrons\nParticle 0: Impact Parameter=%5.3f $\mu$m, $R_{L}$=%5.3f $\mu$m \
                     \nParticle 1: Impact Parameter=%5.3f $\mu$m, $R_{L}$=%5.3f $\mu$m \
                     \nParticle 2: Impact Parameter=%5.3f $\mu$m, $R_{L}$=%5.3f $\mu$m \
                     \nParticle 3: Impact Parameter=%5.3f $\mu$m, $R_{L}$=%5.3f $\mu$m \
                     \nParticle 4: Impact Parameter=%5.3f $\mu$m, $R_{L}$=%5.3f $\mu$m' % \
           (1.e+4*impPar_slctd[0],1.e+4*rhoLarm_slctd[0], \
	    1.e+4*impPar_slctd[0],1.e+4*rhoLarm_slctd[1], \
	    1.e+4*impPar_slctd[0],1.e+4*rhoLarm_slctd[2], \
	    1.e+4*impPar_slctd[0],1.e+4*rhoLarm_slctd[3], \
	    1.e+4*impPar_slctd[0],1.e+4*rhoLarm_slctd[4])), color='m',fontsize=6)
plt.xlabel('x, $\mu m$',color='m',fontsize=16)
plt.ylabel('y, $\mu m$',color='m',fontsize=16)

fig130=plt.figure(130)
plt.plot(range(points),particles[Ipx,0:points,1]/m_elec/rmsV_eTran,'.r')
plt.xlabel('Time Step',color='m',fontsize=16)
plt.ylabel('Vx, cm/sec',color='m',fontsize=16)
plt.grid(True)

fig140=plt.figure(140)
plt.plot(range(points),particles[Ix,0:points,1],'.r')
plt.xlabel('Time Step',color='m',fontsize=16)
plt.ylabel('x, cm',color='m',fontsize=16)
plt.grid(True)

fig150=plt.figure(150)
plt.plot(range(points),particles[Ipy,0:points,1]/m_elec/rmsV_eTran,'.r')
plt.xlabel('Time Step',color='m',fontsize=16)
plt.ylabel('Vy, cm/sec',color='m',fontsize=16)
plt.grid(True)

fig160=plt.figure(160)
plt.plot(range(points),particles[Iy,0:points,1],'.r')
plt.xlabel('Time Step',color='m',fontsize=16)
plt.ylabel('y, cm',color='m',fontsize=16)
plt.grid(True)

plt.show()   

sys.exit()   

#            phi=np.random.uniform(low=0.,high=2.*pi,size=1)
#            z_elec_crnt[Ix]=impPar_slctd[j]*math.cos(phi)                 # x, cm
#            z_elec_crnt[Iy]=impPar_slctd[j]*math.sin(phi)                 # y, cm


#       print 'z_elec[:,0]: ',z_elec[:,0] 
#       type_zFin_elec=isinstance(zFin_elec,list)
#       print 'type_zFin_elec=list: ',type_zFin_elec 
#       type_zFin_elec=isinstance(zFin_elec,tuple)
#       print 'type_zFin_elec=tuple: ',type_zFin_elec 