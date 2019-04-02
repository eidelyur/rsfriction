# from __future__ import division

#-------------------------------------
#
#        Started at 08/11/2017 (YuE)
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

eVtoErg=1.602e-12                                                  # energy from eV to erg (from CI to CGS)
# Indices:
(Ix, Ipx, Iy, Ipy, Iz, Ipz) = range(6)

#
# Initial parameters:
#
Z_ion = qe*2.997e+9                                                # charge of ion (proton), CGSE units of the charge
M_ion = mp*1.e+3                                                   # mass of ion (proton), g
q_elec = qe*2.997e+9                                               # charge of electron, CGSE units of the charge
m_elec = me*1.e+3                                                  # mass of electron, g
tangAlpha=3.                                                       # to calculate length of interaraction
B_mag = 1000.                                                      # magnetic field, Gs
eTempTran = 0.5                                                    # transversal temperature of electrons, eV
eTempLong = 2.e-4                                                  # longitudinal temperature of electrons, eV
numb_e = 1000                                                      # number of electrons
numb_p = 50                                                        # number of protons

eBeamRad = 0.1                                                     # cm
eBeamDens = 1.e+8                                                  # cm^-3
kinEnergy_eBeam=470.*eVtoErg                                       # erg

stepsNumberOnGyro = 40                                             # number of the steps on each Larmour period

#
# Larmor frequency electron:
#
def omega_Larmor(mass,B_mag):
    return (q_elec)*B_mag/(mass*clight*1.e+2)                      # rad/sec

#
# Derived quantities:
#
shiftV_e=np.sqrt(2.*kinEnergy_eBeam/m_elec)                        # cm/sec
#
# The longitudinal shift velocities of the electrons and ions are the same:
# 
kinEnergy_pBeam=kinEnergy_eBeam/m_elec*M_ion                       # erg
shiftV_p=np.sqrt(2.*kinEnergy_pBeam/M_ion)                         # cm/sec
print 'shiftV_e = %e cm/sec, shiftV_p = %e cm/sec' % (shiftV_e,shiftV_p)

tempRatio=eTempLong/eTempTran                                      # dimensionless
velRatio=np.sqrt(tempRatio)                                        # dimensionless
print 'tempRatio = %e, velRatio = %e' % (tempRatio,velRatio)


omega_L = omega_Larmor(m_elec, B_mag)                              # rad/sec 
T_larm = 2*pi/omega_L                                              # sec
timeStep = T_larm/stepsNumberOnGyro                                # time step, sec
print 'omega_Larmor= %e rad/sec, T_larm = %e sec, timeStep = %e sec' % (omega_L,T_larm,timeStep)

eVrmsTran = np.sqrt(2.*eTempTran*eVtoErg/m_elec)                   # cm/sec
eVrmsLong = np.sqrt(2.*eTempLong*eVtoErg/m_elec)                   # cm/sec
print 'eVrmsTran = %e cm/sec, eVrmsLong = %e cm/sec' % (eVrmsTran,eVrmsLong)

# Larmor frequency:
#
ro_larm = eVrmsTran/omega_L                                        # cm
print '<ro_larm> , mkm = ', ro_larm*1.e4

# Plasma frequency of the beam:
#
omega_e=np.sqrt(4*pi*eBeamDens*q_elec**2/m_elec)                   # rad/sec
print 'Plasma''s frequency: omega_e = %e rad/sec' % omega_e

#
# 2D marices for all electrons and ions:
#
z_elec=np.zeros((6,numb_e)) # z_elec(6,:) is a vector: x_e,px_e,y_e,py_e,z_e,pz_e for each electron
z_ion=np.zeros((6,numb_p))  # z_ion(6,:)  is a vector: x_i,px_i,y_i,py_i,z_i,pz_i for each proton

#
# Convertion from electron's "coordinates" to guiding-center coordinates:
# For each electron z_e=(x_e,px_e,y_e,py_e,z_e,pz_e) --> zgc_e=(phi,p_phi,y_gc,py_gc,z_e,pz_e);
# z_c and zgc_e are 2D arrays with dimension (6,n_elec) 
#
def toGuidingCenter(z_e):
    mOmega=m_elec*omega_L                                          # g/sec
    zgc_e=z_e.copy()                                               # 2D array with dimension (6,n_elec)
    zgc_e[Ix,:] = np.arctan2(z_e[Ipx,:]+mOmega*z_e[Iy,:],z_e[Ipy,:])             # radians
    zgc_e[Ipx,:]= (((z_e[Ipx,:]+mOmega*z_e[Iy,:])**2+z_e[Ipy,:]**2)/(2.*mOmega)) # g*cm**2/sec
    zgc_e[Iy,:] =-z_e[Ipx,:]/mOmega                                # cm
    zgc_e[Ipy,:]= z_e[Ipy,:]+mOmega*z_e[Ix,:]                      # g/sec
    return zgc_e

#
# Convertion from guiding-center coordinates to electron's "coordinates":
# For each electron zgc_e=(phi,p_phi,y_gc,py_gc,z_e,pz_e) --> z_e=(x_e,px_e,y_e,py_e,z_e,pz_e);
# zgc_c and z_e are 2D arrays with dimension (6,n_elec) 
#
def fromGuidingCenter(zgc_e):
    mOmega=m_elec*omega_L                                          # g/sec
    rho_larm=np.sqrt(2.*zgc_e[Ipx,:]/mOmega)                       # cm
    z_e = zgc.copy()                                               # 2D array with dimension (6,n_elec)
    z_e[Ix,:] = zgc_e[Ipy,:]/mOmega-rho_larm*np.cos(zgc_e[Ix,:])   # cm
    z_e[Ipx,:]=-mOmega*zgc_e[Iy,:]                                 # g*cm/sec
    z_e[Iy,:] = zgc_e[Iy,:]+rho_larm*np.sin(zgc_e[Ix,:])           # cm
    z_e[Ipy,:]= mOmega*rho_larm*np.cos(zgc_e[Ix,:])                # g*cm/sec
    return z_e

#
# Initial ion (proton) coordinates: 
# all ions are placed in the beginning of the coordinate system
#          and have longitudinal velocity shiftV_p
#
z_ion[Ipz,:]=M_ion*shiftV_p                                        # all ions have the same momenta, g*cm/sec

#
# Matrix to dragg particle with mass 'm_part' through the drift during time interval 'deltaT':
#
def driftMatrix(m_part,deltaT,driftMtrx=[]):
    import numpy as np   # Next np.identity does not work without that! Why?!
    drftMtrx=np.identity(6)
    for i in (Ix,Iy,Iz):
       drftMtrx[i, i + 1]=deltaT/m_part                            # sec/g
    return drftMtrx

#
# Matrix to dragg electron through the solenoid with field 'B_mag' during time interval 'deltaT':
#
def solenoid_eMatrix(B_mag,deltaT):
    import numpy as np   # Next np.identity does not work without that! Why?!
    slndMtrx=np.identity(6)
    omega_L=omega_Larmor(m_elec,B_mag)                             # rad/sec 
    mOmega= m_elec*omega_L                                         # g/sec
    phi=omega_L*deltaT                                             # phase, rad
    cosPhi=math.cos(phi)                                           # dimensionless                                  
    sinPhi=math.sin(phi)                                           # dimensionless
    cosPhi_1=2.*math.sin(phi/2.)**2                                # dimensionless                      
    slndMtrx[Iy, Iy ]= cosPhi                                      # dimensionless
    slndMtrx[Ipy,Ipy]= cosPhi                                      # dimensionless
    slndMtrx[Iy, Ipy]= sinPhi/mOmega                               # sec/g
    slndMtrx[Ipy,Iy ]=-mOmega*sinPhi                               # g/sec
    slndMtrx[Iz, Ipz]= deltaT/m_elec                               # sec/g
    slndMtrx[Ix, Ipx]= sinPhi/mOmega                               # sec/g
    slndMtrx[Ix, Iy ]= sinPhi                                      # dimensionless
    slndMtrx[Ix, Ipy]= cosPhi_1/mOmega                             # sec/g 
    slndMtrx[Iy, Ipx]=-cosPhi_1/mOmega                             # sec/g
    slndMtrx[Ipy,Ipx]=-sinPhi                                      # dimensionless
    return slndMtrx

#
# Dragg electron and ion through the "collision" during time interval 'deltaT':
# During collision momenta of the particles sre changed by value deltaV_int/deltaR^3,
# where deltaV_int=Vion_i-Velec_i (i=x,y,z) and deltaR is a distance between particles
#
def draggCollision(deltaT,z_i,z_e):
    g=deltaT*Z_ion*q_elec**2                                       # g*cm^3/sec
    dz=z_i-z_e
    denom=(dz[Ix]**2+dz[Iy]**2+dz[Iz]**2)**(3/2)                   # cm^3
    zf_i=z_i.copy()
    zf_e=z_e.copy()
    for ip in (Ipx,Ipy,Ipz):
       zf_i[ip]=z_i[ip]-g*dz[ip-1]/denom                           # g*cm/sec
       zf_e[ip]=z_e[ip]+g*dz[ip-1]/denom                           # g*cm/sec
    return zf_i,zf_e

#================================================================
#
#  Integrations to calculate the friction force
#
numbVe_tran=30
numbImpPar=30
numbVe_long=30

#
# It was found that the minimal impact parameter (minImpctPar) is 
# defined by the Larmor frequency only and equals rhoCrit  
# (rhoCrit = 1 mkm for B = 1000 Gs); magnetization of the electron
# means that minimal distance between ion and electron larger than rhoCrit,
# i.e. minImpctPar > rhoCrit+rho_larm  
#
rhoCrit=math.pow(q_elec**2/(m_elec*omega_L**2),1./3)               # cm
print 'rhoCrit, <ro_larm> (mkm) = ', (1.e4*rhoCrit,1.e4*ro_larm)

numb_e = 1000                                                      # number of electrons
numb_p = 50                                                        # number of protons

eBeamRad = 0.1                                                     # cm
eBeamDens = 1.e+8                                                  # cm^-3

neutrRadius=math.pow(.75/eBeamDens,1./3.)                    # cm
print 'Neutalization radius (mkm) = ', 1.e4*neutrRadius
#
# It was found that the maximal impact parameter (maxImpctPar) is defined  
# by the longitudinal temperature eTempLong of electrons and their density
# eBeamDens in the beam; for "small" velocity V_i of the ions the maxImpctPar
# is constant = neutrRadius, while the opposite case it depend linearly of V_i:
# =slopDebye*V_i/eVrmsLong. So, maxImpctPar will calculate in the place,
# where the velocities of the ion will be defined: 
#
# maxImpctPar = radiusDebye = slopDebye * V_i / eVrmsLong 
# if maxImpctPar < neutrRadius:
#    maxImpctPar = neutrRadius 
#
slopDebye=np.sqrt(eVtoErg*eTempLong/(2*pi*q_elec**2*eBeamDens))            # cm
print 'slopDebye (mkm): ', 1.e4*slopDebye

#
# Initial gaussian distributions of the relative transverse electron's velocity 
# in polar velocities coordinates: 
#
eVtran=np.zeros(numb_e)
for i in range(numb_e):
   ksi1=np.random.uniform(low=0.,high=1.,size=1)
   ksi2=np.random.uniform(low=0.,high=1.,size=1)
   z_elec[Ipx,i]=np.sqrt(-2.*math.log(ksi1))*math.cos(2.*pi*ksi2)  # dimensionless: eVx_tran/eVrmsTran
   z_elec[Ipy,i]=np.sqrt(-2.*math.log(ksi1))*math.sin(2.*pi*ksi2)  # dimensionless: eVy_tran/eVrmsTran
   eVtran[i]=np.sqrt(z_elec[Ipx,i]**2+z_elec[Ipy,i]**2)            # dimensionless: eVtran/eVrmsTran

# print 'z_elec[Ipx,:]=',z_elec[Ipx,:]
# print 'z_elec[Ipy,:]=',z_elec[Ipy,:]
eVtranStd=eVtran.std()
print 'Relative: eVtranStd=',eVtranStd

eVtranMin=np.min(eVtran)                                           # dimensionless
eVtranMax=np.max(eVtran)                                           # dimensionless
print 'Relative (transversal): eVtranMin, eVtranMax = ', (eVtranMin,eVtranMax)
print 'Absolute (transversal) cm/sec: eVtranMin, eVtranMax = ', (eVrmsTran*eVtranMin,eVrmsTran*eVtranMax)
 
eVtranStep=(eVtranMax-eVtranMin)/numbVe_tran                       # dimensionless

eVtranSlctd=np.zeros(numbVe_tran)                                  # dimensionless
for k in range(numbVe_tran):
   eVtranSlctd[k]=eVtranMin+eVtranStep*(k+.5)                      # dimensionless

print 'eVtranSlctd = ',eVtranSlctd

plt.figure(10)
velTran_hist=plt.hist(eVtran,bins=30)
plt.xlabel('$V_\perp / V_{rms_\perp}$',color='m',fontsize=16)
plt.ylabel('Particles',color='m',fontsize=16)
plt.ylim([0,1.025*np.max(velTran_hist[0])])
plt.title(('Initial Distribution of $V_\perp / V_{rms_\perp}$ (%d Electrons)'  % numb_e), \
          color='m',fontsize=16)
# plt.text(0,1.025*np.max(velTran_hist[0]),('From Distribution: $V_{rms}$ = %6.4f' % stdVtran), \
# 	 color='m',fontsize=16,ha='center')	  
plt.grid(True)

#
# Initial uniform distribution of the transverse velocity in cross section:
#
phi=np.random.uniform(high=2*pi,size=numb_e)
for i in range(numb_e):
   fi=np.random.uniform(low=0.,high=2*pi,size=1)
   z_elec[Ipx,i]=eVtran[i]*math.cos(fi)                            # dimensionless velocity (!)
   z_elec[Ipy,i]=eVtran[i]*math.sin(fi)                            # dimensionless velocity (!) 

#
# Initial gaussian distribution of the relative longitudinal electron's velocities 
#
relShiftV_e=shiftV_e/eVrmsLong                                     # dimensionless velocity

# dimensionless velocity (!):
z_elec[Ipz,:]=np.random.normal(loc=relShiftV_e,scale=1.,size=numb_e)    # eVz/eVrmsLong 

# Verifying of distribution:
eVzMean=z_elec[Ipz,:].mean()
eVzStd=z_elec[Ipz,:].std()
print 'Relative: eVzMean = %e (must be %e),eVzStd = %e (must be %e)' % (eVzMean,relShiftV_e,eVzStd,1.)

eVlongMin=np.min(abs(z_elec[Ipz,:]-relShiftV_e))                   # dimensionless
eVlongMax=np.max(abs(z_elec[Ipz,:]-relShiftV_e))                   # dimensionless
print 'Relative (longitudinal): eVlongMin, eVlongMax = ', (eVlongMin,eVlongMax)
print 'Absolute (longitudinal), cm/sec: eVlongMin, eVlongMax = ', (eVrmsLong*eVlongMin,eVrmsLong*eVlongMax)
 
plt.figure(20)
velLong_hist=plt.hist(z_elec[Ipz,:],bins=30)
plt.xlabel('$V_z / V_{rms_{||}}$',color='m',fontsize=16)
plt.ylabel('Particles',color='m',fontsize=16)
plt.ylim([0,1.025*np.max(velLong_hist[0])])
plt.title(('Initial Distribution of $V_z / V_{rms_{||}}$ (%d Electrons)'  % numb_e), \
          color='m',fontsize=16)
# plt.text(0,1.025*np.max(velTran_hist[0]),('From Distribution: $V_{rms}$ = %6.4f' % stdVtran), \
# 	 color='m',fontsize=16,ha='center')	  
plt.grid(True)

#
# Initial distribution of the longitudinal velocity:
#
eVlongMin=np.min(z_elec[Ipz,:])                                    # dimensionless
eVlongMax=np.max(z_elec[Ipz,:])                                    # dimensionless
print 'Relative: eVlongMin, eVlongMax = ',(eVlongMin,eVlongMax)

eVlongSlctd=np.zeros(numbVe_long)                                  # dimensionless
eVlongStep=(eVlongMax-eVlongMin)/numbVe_long                       # dimensionless

for k in range(numbVe_long):
   eVlongSlctd[k]=eVlongMin+eVlongStep*(k+.5)                      # dimensionless
# print 'Relative eVlongSlctd =\n', eVlongSlctd
# print 'Absolute eVlongSlctd (cm/sec) =\n' , eVrmsLong*eVlongSlctd-shiftV_e


#
# Specific data for selected values for integration: 
# intrLenSlctd - interaction length, tofSlctd - time of flight,
# timeStepsSlctd - total number of time steps 
#
currntRlarm=np.zeros(numbVe_tran)
minImpctPar=np.zeros(numbVe_tran)
maxImpctPar=np.zeros(numbVe_tran)
impParStep=np.zeros(numbVe_tran)
impParSlctd=np.zeros((numbVe_tran,numbImpPar))
intrLenSlctd=np.zeros((numbVe_tran,numbImpPar))
tofSlctd=np.zeros((numbVe_tran,numbImpPar,numbVe_long))
timeStepsSlctd=np.zeros((numbVe_tran,numbImpPar,numbVe_long))
for i in range(numbVe_tran):
   currntRlarm[i] = eVtranSlctd[i]*eVrmsTran/omega_L               # Larmor radius, cm
   minImpctPar[i]=rhoCrit+currntRlarm[i]                           # cm
   shieldRad = slopDebye*eVtranSlctd[i]*eVrmsLong/eVrmsLong        # Debye radius, cm 
   if shieldRad < neutrRadius:
      shieldRad=neutrRadius                                        # Neutralization radius, cm
   maxImpctPar[i]=shieldRad+currntRlarm[i]                         # max of impact parameter, cm
   impParStep[i]=(maxImpctPar[i]-minImpctPar[i])/numbImpPar        # cm
   for j in range(numbImpPar):
      impParSlctd[i,j]=minImpctPar[i]+impParStep[i]*(j+.5)         # cm
      intrLenSlctd[i,j]=2.*impParSlctd[i,j]*tangAlpha              # length of interaction, cm
      for k in range(numbVe_long):
         velLongCrrnt=eVrmsLong*eVlongSlctd[k]                     # vLong, cm/sec
         tofSlctd[i,j,k]=intrLenSlctd[i,j]/np.abs(velLongCrrnt-shiftV_e)   # time of flight for electron, sec
         timeStepsSlctd[i,j,k]=int(tofSlctd[i,j,k]/timeStep)       # total number of steps

'''
# Output for debugging:
eVtranSlctdMin=np.min(eVtranSlctd)                                 # dimensionless
eVtranSlctdMax=np.max(eVtranSlctd)                                 # dimensionless
eVlongSlctdMin=np.min(abs(eVlongSlctd-relShiftV_e))                # dimensionless
eVlongSlctdMax=np.max(abs(eVlongSlctd-relShiftV_e))                # dimensionless
 
minIntLength=np.min(intrLenSlctd)
maxIntLength=np.max(intrLenSlctd)
indMinLength=np.where(intrLenSlctd==np.min(intrLenSlctd))
indMaxLength=np.where(intrLenSlctd==np.max(intrLenSlctd))
minTimeSteps=np.min(timeStepsSlctd)
maxTimeSteps=np.max(timeStepsSlctd)
indMinSteps=np.where(timeStepsSlctd==np.min(timeStepsSlctd))
indMaxSteps=np.where(timeStepsSlctd==np.max(timeStepsSlctd))
print 'Relative (selected transversal): eVtranMin, eVtranMax = ', (eVtranSlctdMin,eVtranSlctdMax)
print 'Absolute (selected transversal) cm/sec: eVtranMin, eVtranMax = ', \
      (eVrmsTran*eVtranSlctdMin,eVrmsTran*eVtranSlctdMax)
print 'Larmour radius (mkm)= ', 1.e4*currntRlarm 
for i in range(numbImpPar):
   print '  (i,   Transvese velocity (cm/sec))= ', (i,eVtranSlctd[i]*eVrmsLong)
   print 'Impact parameter (mkm): min,max=', (1.e4*minImpctPar[i],1.e4*maxImpctPar[i])
   print 'Array of mpact parameters (mkm) is:\n', 1.e4*impParSlctd[i,:]
print 'Relative (selected longitudinal): eVlongMin, eVlongMax = ', (eVlongSlctdMin,eVlongSlctdMax)
print 'Absolute (selected longitudinal) cm/sec: eVlongMin, eVlongMax = ', \
      (eVrmsLong*eVlongSlctdMin,eVrmsLong*eVlongSlctdMax)
print 'Interaction length: min,max= ', (1.e4*minIntLength,1.e4*maxIntLength)
print 'indMinLength= ',indMinLength
print 'indMaxLength= ',indMaxLength
print 'Total number of time steps: min,max= ', (minTimeSteps,maxTimeSteps)
print 'indMinSteps= ',indMinSteps
print 'indMaxSteps= ',indMaxSteps
for i in range(numbVe_tran):
   print '  (i,   Transvese velocity (cm/sec))= ', (i,eVtranSlctd[i]*eVrmsLong)
   print 'Length of interaction (mkm): \n', 1.e4*intrLenSlctd[i,:]
   for j in range(numbImpPar):
      print '  (i,j: Number of time steps for length of interaction (mkm))=  ', (i,j,1.e4*intrLenSlctd[i,j])
      print 'Numbers are:\n', timeStepsSlctd[i,j,:]
'''	
    
#
# Returning to momenta:
#
z_elec[Ipx,:]=m_elec*eVrmsTran*z_elec[Ipx,:]                       # g*cm/sec
z_elec[Ipy,:]=m_elec*eVrmsTran*z_elec[Ipy,:]                       # g*cm/sec
z_elec[Ipz,:]=m_elec*eVrmsLong*(z_elec[Ipz,:]-relShiftV_e)         # g*cm/sec

#================================================================#
#                                                                #
#  First approach to calculate the friction force                #
#                                                                #
#================================================================#
#
# Preparation for integration:
#
factorIntgrtnInit=np.sqrt(2.*pi)*eVtranStep*eVlongStep             # dimensionless
factorIntgrtnInit *= eBeamDens/(eVrmsTran**2*eVrmsLong)            # sec^3/cm^6
print 'factorIntgrtnInit (1/sec^3)= ', factorIntgrtnInit

# rhoLarmSlctd=np.zeros(numbVe_tran)                               # now: currntRlarm

# For drawing and debugging:
particles=np.zeros((6,10000,numbVe_tran*numbImpPar*numbVe_long))   # to draw the trajectories 
#                                                                    (not more than 10000 points for the each electron)

velIonSlctd=np.zeros(numb_p)                                       # selected velocity of the ion
cpuTime_p=np.zeros(numb_p)                                         # time per ion (proton) 
cpuTime_e=np.zeros(numbVe_tran*numbImpPar*numbVe_long)             # time per electron
elpsdTime_p=np.zeros(numb_p)                                       # time per ion (proton)
elpsdTime_e=np.zeros(numbVe_tran*numbImpPar*numbVe_long)           # time per one electron

deltaPion=np.zeros((3,numb_p))                                     # increasing of the momentum for each ion
dPionCrrnt=np.zeros(3)                                             # increasing of the momentum for current ion
frctnForce=np.zeros((3,numb_p))                                    # friction force for each ion
matr_ion=driftMatrix(M_ion,.5*timeStep)                            # matrix for ion for half of time step (drift gap)
matr_elec=solenoid_eMatrix(B_mag,.5*timeStep)                 # matrix for electron for half of time step (magnetic field)

'''
print 'matr_ion: ',matr_ion
print 'matr_elec: ',matr_elec
'''

#
# Integration along the larmour trajectories:
#
gFactor=timeStep*Z_ion*q_elec**2                                   # g*cm^3/sec
z_elecCrrnt=np.zeros(6)                                            # Initial vector for electron
z_ionCrrnt=np.zeros(6)                                             # Initial vector for ion
#
# Exceed the ion velocity over the longitudinal rms velocity of the velectron:
#
exceedFctr=[5,10,20]
eVtranRange=(numbVe_tran)
impParRange=(numbImpPar)
eVlongRange=(numbVe_long)

# For debugging:
eVtranRange=10
impParRange=10
eVlongRange=10

for nion in range(1):                       # range(numb_p):
#
# Selection of the ion velocity
#
   velIonSlctd[nion]=exceedFctr[nion]*eVrmsLong                    # Selected velocity of the ion, cm/sec
   velIonCrrnt=exceedFctr[nion]*eVrmsLong                          # Current velocity of the ion, cm/sec
   for m in range(6):
      z_ionCrrnt[m]=0.                                             # Initial zero-vector for ion
   z_ionCrrnt[Ipz]=M_ion*velIonSlctd[nion]                         # Current pz of ion, g*cm/sec                        
   for i in range(eVtranRange):                                    # Integration over transversal velocity
      velTranCrrnt=eVrmsTran*eVtranSlctd[i]                        # Current vTran, cm/sec
      for j in range(impParRange):                                 # Integration over impact parameter
#         factorIntgrtnCrrnt=impParSlctd[i,j]                       # cm
         intrcnLength=intrLenSlctd[i,j]                            # Current length of interaction, cm
         for k in range(eVlongRange):                              # Integration over longitudinal velocity
	    timeStart=os.times()
#	    print '     TimeStart: ', timeStart
 	    numbCrrntElec=impParRange*(eVtranRange*i+j)+k          # Current number (tag) of the current electron
            velLongCrrnt=eVrmsLong*eVlongSlctd[k]                  # Current vLong, cm/sec
	    absDeltaV=np.sqrt(velTranCrrnt**2+(velIonCrrnt-velLongCrrnt)**2)    # |Vion-Velec|, cm/sec
#	     factorIntgrtnCrrnt *= absDeltaV                        # cm*(cm/sec)**2
# 	     factorIntgrtnCrrnt *= velTranCrrnt                     # cm*cm/sec
#            print 'factorIntgrtnCrrnt %e ' % factorIntgrtnCrrnt 
            timeOfFight=tofSlctd[i,j,k]                            # Current time of flight for the electron, sec
            timeSteps=int(timeStepsSlctd[i,j,k])                   # Current total number of the steps
            for m in range(6):
               z_elecCrrnt[m]=0.                                   # Initial zero-vector for electron
#
# Initial position of electrons (x=impactParameter, y=0, ...):
#
            z_elecCrrnt[Ix]=impParSlctd[i,j]                       # x, cm
            z_elecCrrnt[Iz]=-.5*intrcnLength                       # z, cm
#            phi=np.random.uniform(low=0.,high=2.*pi,size=1)
#            z_elecCrrnt[Ipx]=m_elec*velTranCrrnt*math.cos(phi)     # px, g*cm/sec
#            z_elecCrrnt[Ipy]=m_elec*velTranCrrnt*math.sin(phi)     # py, g*cm/sec
            z_elecCrrnt[Ipy]=m_elec*velTranCrrnt                   # py, g*cm/sec
            z_elecCrrnt[Ipz]=m_elec*(velLongCrrnt-shiftV_e)        # pz, g*cm/sec
            zFin_ion=[z_ionCrrnt]
            zFin_elec=[z_elecCrrnt]
            zFin_ion.append(z_ionCrrnt)
            zFin_elec.append(z_elecCrrnt)
#
# Dragging  electron near each protons:
#
#             print 'Electron %d: steps = %d, rhoLarm(mkm) = %8.6f' % \
#                   (numbCrrntElec,timeSteps,1.e+4*currntRlarm[i])
            pointTrack=0
            for istep in range(timeSteps):
#
# Before interaction:
#	       
               z_ionCrrnt=matr_ion.dot(z_ionCrrnt)
 	       z_elecCrrnt=matr_elec.dot(z_elecCrrnt)
# To draw an ion's and electron's trajectories for checking:
               if numbCrrntElec==0:	       
                  particles[:,pointTrack,0]=z_ionCrrnt
               particles[:,pointTrack,numbCrrntElec+1]=z_elecCrrnt
               pointTrack += 1
#----------------	       
#
# Interaction between ion and electron:
#	       
###               z_ionCrrnt,z_elecCrrnt=draggCollision(timeStep,z_ionCrrnt,z_elecCrrnt)
               dz=z_ionCrrnt-z_elecCrrnt
               denom=(dz[Ix]**2+dz[Iy]**2+dz[Iz]**2)**(3/2)        # cm^3
               for ip in (Ipx,Ipy,Ipz):
                  dPionCrrnt[ip//2] = -gFactor*dz[ip-1]/denom      # g*cm/sec
                  z_ionCrrnt[ip] =z_ionCrrnt[ip] +dPionCrrnt[ip//2]        # g*cm/sec
                  z_elecCrrnt[ip]=z_elecCrrnt[ip]-dPionCrrnt[ip//2]        # g*cm/sec
                  deltaPion[ip//2,nion] += dPionCrrnt[ip//2]       # g*cm/sec
#                  print 'deltaPion (step %d): %e %e %e ' % \
#                        (istep,deltaPion[0,nion],deltaPion[1,nion],deltaPion[2,nion])
#
#----------------
# To draw an ion's and electron's trajectories for checking:
               if numbCrrntElec==0:	       
                  particles[:,pointTrack,0]=z_ionCrrnt
               particles[:,pointTrack,numbCrrntElec+1]=z_elecCrrnt
               pointTrack += 1
#
# After interaction:
#	       
               z_ionCrrnt=matr_ion.dot(z_ionCrrnt)
 	       z_elecCrrnt=matr_elec.dot(z_elecCrrnt)
# To draw an ion's and electron's trajectories for checking:
               if numbCrrntElec==0:	       
                  particles[:,pointTrack,0]=z_ionCrrnt
               particles[:,pointTrack,numbCrrntElec+1]=z_elecCrrnt
               pointTrack += 1
               zFin_ion.append(z_ionCrrnt)
               zFin_elec.append(z_elecCrrnt)
#	       frctnForce[0,nion] += factorIntgrtnCrrnt*deltaPx_ion        # g*cm*(cm/sec)**3
#	       frctnForce[1,nion] += factorIntgrtnCrrnt*deltaPy_ion        # g*cm*(cm/sec)**3
#	       frctnForce[2,nion] += factorIntgrtnCrrnt*deltaPz_ion        # g*cm*(cm/sec)**3
#                print 'Friction force (step %d): %e %e %e ' % \
# 	             (istep,frctnForce[0,nion],frctnForce[1,nion],frctnForce[2,nion])
#            print 'Friction force: %e %e %e ' % (frctnForce[0,nion],frctnForce[1,nion],frctnForce[2,nion])
            timeEnd=os.times()
#            print '                  TimeEnd: ', timeEnd
            cpuTime_e[numbCrrntElec]   = float(timeEnd[0])-float(timeStart[0])   # CPU time for the current electron
            elpsdTime_e[numbCrrntElec] = float(timeEnd[4])-float(timeStart[4])   # Elapsed real time for the current electron
            cpuTime_p[nion]   += cpuTime_e[numbCrrntElec]                         # CPU time for the current proton
            elpsdTime_p[nion] += elpsdTime_e[numbCrrntElec]                       # Elapsed real time for the current proton
###   frctnForce[0,nion] = frctnForce[0,nion]*factorIntgrtnInit            # g*cm/sec**2
###   frctnForce[1,nion] = frctnForce[1,nion]*factorIntgrtnInit            # g*cm/sec**2
###   frctnForce[2,nion] = frctnForce[2,nion]*factorIntgrtnInit            # g*cm/sec**2

# For debugging:
   minCPUperStep=1.e10
   maxCPUperStep=0
   for i in range(eVtranRange):                                    # Integration over transversal velocity
      for j in range(impParRange):                                 # Integration over impact parameter
         for k in range(eVlongRange):                              # Integration over longitudinal velocity
 	    numbCrrntElec=impParRange*(eVtranRange*i+j)+k          # Current number (tag) of the current electron
            timeSteps=int(timeStepsSlctd[i,j,k])                   # Current total number of the steps
	    cpuTimePerStep=1.e+6*elpsdTime_e[numbCrrntElec]/timeSteps      # mks 
	    if minCPUperStep > cpuTimePerStep:
	       if cpuTimePerStep > 0.:
	          minCPUperStep=cpuTimePerStep 
	    if maxCPUperStep < cpuTimePerStep:
	       maxCPUperStep=cpuTimePerStep 
            print 'Electron %d: steps = %d, cpu(mks) = %e, elapsed(mks) = %e, cpuPerStep(mks) = %e' % \
                  (numbCrrntElec,timeSteps,1.e+6*cpuTime_e[numbCrrntElec],1.e+6*elpsdTime_e[numbCrrntElec], \
		   cpuTimePerStep)
   print ' Time per Step (mks): min=%6.3f, max=%6.3f' % (minCPUperStep,maxCPUperStep)		   
   print '        Proton %d: electrons = %d, cpu(s) = %e, elapsed(s) = %e' % \
         (nion,numbCrrntElec+1,cpuTime_p[nion],elpsdTime_p[nion])
   print 'deltaPion: (ion %d) %e %e %e ' % (nion,deltaPion[0,nion],deltaPion[1,nion],deltaPion[2,nion])


'''	
#----------------------------------------------
#              Nondebugged part:
#---------------------------------------------- 
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
'''	

plt.show()   

sys.exit()   
