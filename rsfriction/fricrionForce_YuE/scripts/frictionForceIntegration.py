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

eVtoErg=1.602e-12                # energy from eV to erg (from CI to CGS)
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

eBeamRad = 0.1                   # cm
eBeamDens = 1.e+8                # cm^-3
kinEnergy_eBeam=470.*eVtoErg     # erg

stepsNumberOnGyro = 40           # number of the steps on each Larmour period

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
print 'omega_e = %e rad/sec' % omega_e

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
z_ion[Ipz,:]=M_ion*shiftV_p           # all ions have the same momenta, g*cm/sec

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
numbImpPar=30
numbVe_tran=30
numbVe_long=30

#
# It was found that the minimal impact parameter (minImpctPar) is 
# defined by the Larmor frequency only and equals rhoCrit  
# (rhoCrit = 1 mkm for B = 1000 Gs); magnetization of the electron
# means that minimal distance between ion and electron larger than rhoCrit,
# i.e. minImpctPar > rhoCrit+rho_larm  
#
rhoCrit=math.pow(q_elec**2/(m_elec*omega_L**2),1./3)               # cm
# print 'rhoCrit, ro_larm (mkm) = ', (1.e4*rhoCrit,1.e4*ro_larm)

minImpctPar=rhoCrit+ro_larm

numb_e = 1000                                                      # number of electrons
numb_p = 50                                                        # number of protons

eBeamRad = 0.1                                                     # cm
eBeamDens = 1.e+8                                                  # cm^-3
#
# It was found that the maximal impact parameter (maxImpctPar) is defined  
# by the longitudinal temperature eTempLong of electrons and their density
# eBeamDens in the beam; for "small" velocity V_i of the ions the maxImpctPar
# is constant =slopDebye, while the opposite case it depend linearly of V_i:
# =slopDebye*V_i/eVrmsLong. So maxImpctPar will calculate in the place,,
# where the velocities of the ion will be defined: 
#
# maxImpctPar = slopDebye * V_i / eVrmsLong 
# if V_i < eVrmsLong:
#    maxImpctPar = slopDebye 
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
print 'Relative: eVtranMin, eVtranMax = ', (eVtranMin,eVtranMax)
 
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
# timeStepSlctd - total number of time steps 
#
impParSlctd=np.zeros((numbVe_tran,numbImpPar))
intrLenSlctd=np.zeros((numbVe_tran,numbImpPar))
tofSlctd=np.zeros((numbVe_tran,numbImpPar,numbVe_long))
timeStepSlctd=np.zeros((numbVe_tran,numbImpPar,numbVe_long))
for i in range(numbVe_tran):
   maxImpctPar = slopDebye*eVtranSlctd[i]*eVrmsLong/eVrmsLong      # cm 
   if eVtranSlctd[i]*eVrmsLong < eVrmsLong:
      maxImpctPar = slopDebye                                      # cm
   impParStep=(maxImpctPar-minImpctPar)/numbImpPar                 # cm
   for j in range(numbImpPar):
      impParSlctd[i,j]=minImpctPar+impParStep*(j+.5)               # cm
      intrLenSlctd[i,j]=2.*impParSlctd[i,j]*tangAlpha              # length of interaction, cm
      for k in range(numbVe_long):
         velLongCrnt=eVrmsLong*eVlongSlctd[k]                      # vLong, cm/sec
         tofSlctd[i,j,k]=intrLenSlctd[i,j]/np.abs(velLongCrnt-shiftV_e)   # time of flight for electron, sec
         timeStepSlctd[i,j,k]=int(tofSlctd[i,j,k]/timeStep)        # total number of steps

for i in range(numbVe_tran):
   print '               Transvese velocity (cm/sec) ', eVtranSlctd[i]*eVrmsLong
   print 'Length of interaction (mkm): \n', 1.e4*intrLenSlctd[i,:]

'''	
#    for j in range(numbImpPar):
#       print 'Length of interaction (mkm): ', 1.e4*intrLenSlctd[i,j]
#       print 'Number of Larmor turns during interaction:\n', tofSlctd[i,j,:]/T_larm

for j in range(numbImpPar):
   print 'Total number of time steps for length interaction (mkm)', 1.e4*intrLenSlctd[j]
   print 'Numbers are:\n', timeStepSlctd[j,:]
'''	
    
'''	
#----------------------------------------------
#              Nondebugged part:
#---------------------------------------------- 
#
# Returning to momenta:
#
z_elec[Ipx,:]=m_elec*eVrmsTran*z_elec[Ipx,:]                       # g*cm/sec
z_elec[Ipy,:]=m_elec*eVrmsTran*z_elec[Ipy,:]                       # g*cm/sec
z_elec[Ipz,:]=m_elec*eVrmsLong*(z_elec[Ipz,:]-relShiftV_e)         # g*cm/sec

#
# Preparation for integration:
#
factorIntgrtnInit=np.sqrt(2.*pi)*impParStep*eVtranStep*eVlongStep # cm*(cm/sec)**2
factorIntgrtnInit *= eBeamDens/(eVrmsTran**2*eVrmsLong)           # sec/cm**3
print 'factorIntgrtnInit (sec/cm^3)= ', factorIntgrtnInit

rhoLarmSlctd=np.zeros(numbVe_tran)
particles=np.zeros((6,10000,numbImpPar))                           # to draw the trajectories
cpuTime_p=np.zeros(numb_p)
cpuTime_e=np.zeros(numbImpPar)
elpsdTime_p=np.zeros(numb_p)
elpsdTime_e=np.zeros(numbImpPar)

deltaPion=np.zeros((3,numb_p))
dPionCrrnt=np.zeros(3)
frctnForce=np.zeros((3,numb_p))
matr_ion=driftMatrix(M_ion,.5*timeStep)
print 'matr_ion: ',matr_ion
matr_elec=solenoid_eMatrix(B_mag,.5*timeStep)
print 'matr_elec: ',matr_elec

#
# Integration along the larmour trajectories:
#
gFactor=timeStep*Z_ion*q_elec**2                                   # g*cm^3/sec
z_elecCrrnt=np.zeros(6)                                            # Initial vector for electron
z_ionCrrnt=np.zeros(6)                                             # Initial vector for ion
for nion in range(1):                       # range(numb_p):
   for m in range(6):
      z_ionCrrnt[m]=0.                                             # Initial zero-vector for ion
# All ions have the same longitudinal momenta:                        
#    z_ionCrrnt[Ipz]=M_ion*shiftV_p                                # pz, g*cm/sec                                                              
   for j in range(1):                    # range(binsImpPar):
      factorIntgrtnCrrnt=impPar_slctd[j]                           # cm
      intrcnLength=intrLen_slctd[j]                                # length of interaction, cm
      for k in range(1):                  # range(binsVe_long):
         velLongCrnt=rmsV_eLong*velElong_slctd[k]                  # vLong, cm/sec
         print 'relShiftV_e = %e, velElong_slctd[k] = %e, shiftV_p = %e, shiftV_e = %e, velLongCrnt = %e: ' % \
	       (relShiftV_e,velElong_slctd[k],shiftV_p,shiftV_e,velLongCrnt)
         timeOfLight=tol_slctd[j,k]                                # time of flight for electron, sec
         timeSteps=int(tmStep_slctd[j,k])                          # total number of steps
         print 'tmStep_slctd[j,k], timeSteps: ',(tmStep_slctd[j,k],timeSteps)
         for i in range(5):                  # range(binsVe_tran):
	    timeStart=os.times()
            for m in range(6):
               z_elecCrrnt[m]=0.                                   # Initial zero-vector for electron
#
# Initial position of electrons: x=impactParameter, y=0:
#
            z_elecCrrnt[Ix]=impPar_slctd[j]                               # x, cm
            z_elecCrrnt[Iz]=-.5*intrcnLength                              # z, cm
 	    numbCrntElec=binsVe_long*(binsVe_tran*j+k)+i
            rhoLarm_slctd[i]=rmsV_eTran*velEtran_slctd[i]/Omega_e          # cm          
            velTranCrnt=rmsV_eTran*velEtran_slctd[i]                       # vTran, cm/sec
	    factorIntgrtnCrrnt *= velTranCrnt                              # cm*cm/sec
#            print 'factorIntgrtnCrrnt %e ' % factorIntgrtnCrrnt 
	    absDeltaV=np.sqrt(velTranCrnt**2+(shiftV_p-velLongCrnt)**2)    # |Vion-Velec|, cm/sec
	    factorIntgrtnCrrnt *= absDeltaV                                # cm*(cm/sec)**2
# For checking of the trajectories:
#            z_elecCrrnt[Ipx]=0.                                            # px, g*cm/sec
#            z_elecCrrnt[Ipy]=m_elec*velTranCrnt                            # py, g*cm/sec
            phi=np.random.uniform(low=0.,high=2.*pi,size=1)
            z_elecCrrnt[Ipx]=m_elec*velTranCrnt*math.cos(phi)              # px, g*cm/sec
            z_elecCrrnt[Ipy]=m_elec*velTranCrnt*math.sin(phi)              # py, g*cm/sec
            z_elecCrrnt[Ipz]=m_elec*(velLongCrnt-shiftV_e)                            # pz, g*cm/sec
            zFin_ion=[z_ionCrrnt]
            zFin_elec=[z_elecCrrnt]
            zFin_ion.append(z_ionCrrnt)
            zFin_elec.append(z_elecCrrnt)
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
               z_ionCrrnt=matr_ion.dot(z_ionCrrnt)
 	       z_elecCrrnt=matr_elec.dot(z_elecCrrnt)
# To draw ion and first 4 electron trajectories for checking:
#                particles[:,pointTrack,numbCrntElec+1]=zFin_elec[istep+1]
               particles[:,pointTrack,numbCrntElec+1]=z_elecCrrnt
               if numbCrntElec==0:	       
                  particles[:,pointTrack,0]=z_ionCrrnt
               pointTrack += 1
#----------------	       
#
# Interaction between ion and electron:
#	       
###               z_ionCrrnt,z_elecCrrnt=draggCollision(timeStep,z_ionCrrnt,z_elecCrrnt)
               dz=z_ionCrrnt-z_elecCrrnt
               denom=(dz[Ix]**2+dz[Iy]**2+dz[Iz]**2)**(3/2)                # cm^3
               for ip in (Ipx,Ipy,Ipz):
                  dPionCrrnt[ip//2] = -gFactor*dz[ip-1]/denom              # g*cm/sec
                  z_ionCrrnt[ip] =z_ionCrrnt[ip] +dPionCrrnt[ip//2]        # g*cm/sec
                  z_elecCrrnt[ip]=z_elecCrrnt[ip]-dPionCrrnt[ip//2]        # g*cm/sec
                  deltaPion[ip//2,nion] += dPionCrrnt[ip//2]               # g*cm/sec
#
#----------------
# To draw ion and first 4 electron trajectories for checking:
#                particles[:,pointTrack,numbCrntElec+1]=zFin_elec[istep+1]
               particles[:,pointTrack,numbCrntElec+1]=z_elecCrrnt
               if numbCrntElec==0:	       
                  particles[:,pointTrack,0]=z_ionCrrnt
               pointTrack += 1
#
# After interaction:
#	       
               z_ionCrrnt=matr_ion.dot(z_ionCrrnt)
 	       z_elecCrrnt=matr_elec.dot(z_elecCrrnt)
# To draw ion and first 4 electron trajectories for checking:
#                particles[:,pointTrack,numbCrntElec+1]=zFin_elec[istep+1]
               particles[:,pointTrack,numbCrntElec+1]=z_elecCrrnt
               if numbCrntElec==0:	       
                  particles[:,pointTrack,0]=z_ionCrrnt
               pointTrack += 1
###               deltaPx_ion = z_ionCrrnt[Ipx]-z_ion[Ipx,nion]               # deltaPx, g*cm/sec
###               deltaPy_ion = z_ionCrrnt[Ipy]-z_ion[Ipy,nion]               # deltaPy, g*cm/sec
###               deltaPz_ion = z_ionCrrnt[Ipz]-z_ion[Ipz,nion]               # deltaPz, g*cm/sec
#                print 'deltaPion (step %d): %e %e %e ' % (istep,deltaPx_ion,deltaPy_ion,deltaPz_ion)
               zFin_ion.append(z_ionCrrnt)
               zFin_elec.append(z_elecCrrnt)
# To draw ion and first 4 electron trajectories for checking:
#                particles[:,istep,numbCrntElec+1]=zFin_elec[istep+1]
#                if numbCrntElec==0:	       
#                   particles[:,istep,0]=zFin_ion[istep+1]
###	       frctnForce[0,nion] += factorIntgrtnCrrnt*deltaPx_ion        # g*cm*(cm/sec)**3
###	       frctnForce[1,nion] += factorIntgrtnCrrnt*deltaPy_ion        # g*cm*(cm/sec)**3
###	       frctnForce[2,nion] += factorIntgrtnCrrnt*deltaPz_ion        # g*cm*(cm/sec)**3
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
###   frctnForce[0,nion] = frctnForce[0,nion]*factorIntgrtnInit                    # g*cm/sec**2
###   frctnForce[1,nion] = frctnForce[1,nion]*factorIntgrtnInit                    # g*cm/sec**2
###   frctnForce[2,nion] = frctnForce[2,nion]*factorIntgrtnInit                    # g*cm/sec**2
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
'''	

plt.show()   

sys.exit()   



