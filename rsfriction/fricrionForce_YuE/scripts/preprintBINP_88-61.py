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

#----------------------------------------
# Figures 3,4 from preprint
#----------------------------------------

roMax_roMin=[1.e4,1.e3,1.e2,1.e1]
Clog=np.zeros(4)

nTheta=500
thetaMax=90.     # degree
thetaDeg=np.zeros(nTheta)
theta=np.zeros(nTheta)

fLong_poz=np.zeros((nTheta,4))
fTran_poz=np.zeros((nTheta,4))
fLong_neg=np.zeros((nTheta,4))
fTran_neg=np.zeros((nTheta,4))

# print 'fLong_poz=',fLong_poz[1:10,3]


for i in range(nTheta):
   thetaDeg[i]=thetaMax/nTheta*i
   theta[i]=pi/180.*thetaDeg[i]
   for k in range(4):
      Clog[k]=math.log(roMax_roMin[k])
      fLong_poz[i,k]=Clog[k]*math.sin(theta[i])**2 
      fTran_poz[i,k]=-2*Clog[k]*math.sin(theta[i])*math.cos(theta[i])
      fLong_neg[i,k]=fLong_poz[i,k]+4*math.cos(theta[i])**4 
      fTran_neg[i,k]=fTran_poz[i,k]



# Figures 3-lower from preprint

fig100=plt.figure(100)
line1, = plt.plot(thetaDeg,fLong_neg[:,0],'-r',linewidth=2,label='$ro_{max}/ro_{min}=10^4$')   
line2, = plt.plot(thetaDeg,fLong_neg[:,1],'-b',linewidth=2,label='$ro_{max}/ro_{min}=10^3$')   
line3, = plt.plot(thetaDeg,fLong_neg[:,2],'-m',linewidth=2,label='$ro_{max}/ro_{min}=10^2$')   
line4, = plt.plot(thetaDeg,fLong_neg[:,3],'-k',linewidth=2,label='$ro_{max}/ro_{min}=10^1$')   
plt.title('Longitudinal Force Friction for Negative Charge', color='m',fontsize=20)
plt.xlabel('$\Theta$, Degree',color='m',fontsize=16)
plt.ylabel('Value Proporsed to Friction Force, rel.units',color='m',fontsize=16)
plt.legend(handler_map={line1: HandlerLine2D()},loc='upper left')
plt.grid(True)
   
# Figures 3-upper from preprint

fig200=plt.figure(200)
line1, = plt.plot(thetaDeg,fTran_neg[:,0],'-r',linewidth=2,label='$ro_{max}/ro_{min}=10^4$')   
line2, = plt.plot(thetaDeg,fTran_neg[:,1],'-b',linewidth=2,label='$ro_{max}/ro_{min}=10^3$')   
line3, = plt.plot(thetaDeg,fTran_neg[:,2],'-m',linewidth=2,label='$ro_{max}/ro_{min}=10^2$')   
line4, = plt.plot(thetaDeg,fTran_neg[:,3],'-k',linewidth=2,label='$ro_{max}/ro_{min}=10^1$')   
plt.title('Transversal Force Friction for Negative Charge', color='m',fontsize=20)
plt.xlabel('$\Theta$, Degree',color='m',fontsize=16)
plt.ylabel('Value Proporsed to Friction Force, rel.units',color='m',fontsize=16)
plt.legend(handler_map={line1: HandlerLine2D()},loc='lower right')
plt.grid(True)
   
# Figures 4-lower from preprint

fig300=plt.figure(300)
line1, = plt.plot(thetaDeg,fLong_poz[:,0],'-r',linewidth=2,label='$ro_{max}/ro_{min}=10^4$')   
line2, = plt.plot(thetaDeg,fLong_poz[:,1],'-b',linewidth=2,label='$ro_{max}/ro_{min}=10^3$')   
line3, = plt.plot(thetaDeg,fLong_poz[:,2],'-m',linewidth=2,label='$ro_{max}/ro_{min}=10^2$')   
line4, = plt.plot(thetaDeg,fLong_poz[:,3],'-k',linewidth=2,label='$ro_{max}/ro_{min}=10^1$')   
plt.title('Longitudinal Force Friction for Pozitive Charge', color='m',fontsize=20)
plt.xlabel('$\Theta$, Degree',color='m',fontsize=16)
plt.ylabel('Value Proporsed to Friction Force, rel.units',color='m',fontsize=16)
plt.legend(handler_map={line1: HandlerLine2D()},loc='upper left')
plt.grid(True)
   
# Figures 4-upper from preprint

fig400=plt.figure(400)
line1, = plt.plot(thetaDeg,fTran_poz[:,0],'-r',linewidth=2,label='$ro_{max}/ro_{min}=10^4$')   
line2, = plt.plot(thetaDeg,fTran_poz[:,1],'-b',linewidth=2,label='$ro_{max}/ro_{min}=10^3$')   
line3, = plt.plot(thetaDeg,fTran_poz[:,2],'-m',linewidth=2,label='$ro_{max}/ro_{min}=10^2$')   
line4, = plt.plot(thetaDeg,fTran_poz[:,3],'-k',linewidth=2,label='$ro_{max}/ro_{min}=10^1$')   
plt.title('Transversal Force Friction for Pozitive Charge', color='m',fontsize=20)
plt.xlabel('$\Theta$, Degree',color='m',fontsize=16)
plt.ylabel('Value Proporsed to Friction Force, rel.units',color='m',fontsize=16)
plt.legend(handler_map={line1: HandlerLine2D()},loc='lower right')
plt.grid(True)
   
#----------------------------------------
# Figure 1 from preprint (new approach)
#----------------------------------------

nRelVtot=1000
maxRelVtot=3
minRelVtot=.1
theta=0.

relVtot=np.zeros(nRelVtot)
fricLongL=np.zeros(nRelVtot)
fricTranL=np.zeros(nRelVtot)
fricLong=np.zeros(nRelVtot)
fricTran=np.zeros(nRelVtot)
fricTranM=np.zeros(nRelVtot)                             # preprint: (2.16) with V_{p||}=0

ratioTtran_Tlong=300.

def intgrndTranL(t,ratio,Vp,theta):
   nom=np.exp(-(t*Vp*math.sin(theta))**2/(1+t**2)-(t*Vp*math.cos(theta))**2/(1+t**2/ratio))*t**2
   denom=(1+t**2)**2*np.sqrt(1+t**2/ratio)
   intgrnd=nom/denom
#    print '           t=%e, intgrndLong=%e' % (t,intgrnd)
   return intgrnd

def intgrndLongL(t,ratio,Vp,theta):
   nom=np.exp(-(t*Vp*math.sin(theta))**2/(1+t**2)-(t*Vp*math.cos(theta))**2/(1+t**2/ratio))*t**2
   denom=(1+t**2)*np.sqrt(1+t**2/ratio)**3
   intgrnd=nom/denom
#    print '           t=%e, intgrndTran=%e' % (t,intgrnd)
   return intgrnd

def intTranL(ratio,Vp,theta):
   intgrl0=quad(lambda x: intgrndTranL(x,ratio,Vp,theta), 0., np.inf,epsabs=1.e-15,epsrel=1.e-8)[0]
   intgrl=2.*np.sqrt(2/pi)*Vp*intgrl0
#    print '    Tran: intgrl0=%e, intgrl=%e' % (intgrl0,intgrl)   
   return intgrl

def intLongL(ratio,Vp,theta):
   intgrl0=quad(lambda x: intgrndLongL(x,ratio,Vp,theta), 0., np.inf,epsabs=1.e-15,epsrel=1.e-8)[0]
   intgrl=2.*np.sqrt(2/pi)*Vp*intgrl0
#    print '    Tran: intgrl0=%e, intgrl=%e' % (intgrl0,intgrl)   
   return intgrl

def intgrndTran(t,ratio,Vp):
   nom=np.exp(-t**2*Vp**2/(1+t**2))*t**2
   denom=(1+t**2)**2*np.sqrt(1+t**2/ratio)
   intgrnd=nom/denom
#    print '           t=%e, intgrndLong=%e' % (t,intgrnd)
   return intgrnd

def intgrndLong(t,ratio,Vp):
   nom=np.exp(-t**2*Vp**2/(1+t**2/ratio))*t**2
   denom=(1+t**2)*np.sqrt(1+t**2/ratio)**3
   intgrnd=nom/denom
#    print '           t=%e, intgrndTran=%e' % (t,intgrnd)
   return intgrnd

def intTran(ratio,Vp):
   intgrl0=quad(lambda x: intgrndTran(x,ratio,Vp), 0., np.inf,epsabs=1.e-15,epsrel=1.e-8)[0]
   intgrl=2.*np.sqrt(2/pi)*Vp*intgrl0
#    print '    Tran: intgrl0=%e, intgrl=%e' % (intgrl0,intgrl)   
   return intgrl

def intLong(ratio,Vp):
   intgrl0=quad(lambda x: intgrndLong(x,ratio,Vp), 0., np.inf,epsabs=1.e-15,epsrel=1.e-8)[0]
   intgrl=2.*np.sqrt(2/pi)*Vp*intgrl0
#    print '    Tran: intgrl0=%e, intgrl=%e' % (intgrl0,intgrl)   
   return intgrl

totalCPU=0 
totalELPSD=0

for i in range(nRelVtot):
   timeStart=os.times()
   relVtot[i]=minRelVtot+(maxRelVtot-minRelVtot)/nRelVtot*i
   fricLong[i]=intLong(ratioTtran_Tlong,relVtot[i]) 
   fricTran[i]=intTran(ratioTtran_Tlong,relVtot[i]) 
   fricTranM[i]=.5/relVtot[i]**2                             # preprint: (2.16) with V_{p||}=0
   fricLongL[i]=intLongL(ratioTtran_Tlong,relVtot[i],theta) 
   fricTranL[i]=intTranL(ratioTtran_Tlong,relVtot[i],theta) 
   timeEnd=os.times()
   deltaCPU=float(timeEnd[0])-float(timeStart[0])     # CPU time
   deltaELPSD=float(timeEnd[4])-float(timeStart[4])   # elapsed real time
   totalCPU +=deltaCPU
   totalELPSD +=deltaELPSD
#    print 'i=%d, Vtot=%e, Long=%e, Tran=%e, CPU=%e, ELPSD=%e' % \
#          (i,relVtot[i],fricLong[i],fricTran[i],deltaCPU,deltaELPSD)

print '\ntotalCPU=%e, totalELPSD=%e' % (totalCPU, totalELPSD)  

fig600=plt.figure(600)
line1, = plt.semilogx(relVtot,fricLong,'-r',linewidth=2,label='$F_{||}$ (Longitudinal)')   
line2, = plt.semilogx(relVtot,fricTran,'-b',linewidth=2,label='$F_\perp$ (Transversal)')   
# line3, = plt.semilogx(relVtot,fricTranM,'-m',linewidth=2,label='Transversal')   
# line4, = plt.semilogx(relVtot,fricLongL,'--r',linewidth=2,label='Longitudinal(S)')   
# line5, = plt.semilogx(relVtot,fricTranL,'--b',linewidth=2,label='Transversal(S)')   
plt.title('Friction Force F', color='m',fontsize=16)
plt.xlabel('$V_p/v_{e\perp}$',color='m',fontsize=16)
plt.ylabel('$F/(4\pi \cdot n_e \cdot e^4 \cdot Z^2 /(m_e \cdot v_{e\perp}^2))$',color='m',fontsize=16)
plt.xlim([minRelVtot,maxRelVtot])
plt.legend(handler_map={line1: HandlerLine2D()},loc='upper right')
plt.grid(True)


plt.show()   

fig100.savefig('longFrictn_negCharge_fig100.jpg')    
fig200.savefig('transvFrictn_negCharge_fig200.jpg') 
fig300.savefig('longFrictn_posCharge_fig300.jpg')
fig400.savefig('transvFrictn_posCharge_fig400.jpg') 
fig600.savefig('frictionForces_fig600.jpg')

sys.exit()   

#----------------------------------------
# Figure 1 from preprint (old approach)
#----------------------------------------

nRelVtot=5
maxRelVtot=.005
minRelVtot=.001

relVtot=np.zeros(nRelVtot)
fricLong=np.zeros(nRelVtot)
fricTran=np.zeros(nRelVtot)
fricMagn=np.zeros(nRelVtot)

ratioTtran_Tlong=300.
maxVeTran=3.
maxVeLong=maxVeTran/np.sqrt(ratioTtran_Tlong)

def intgrndLong(x,y,ratio,p):
   nom=(p-y)*np.exp(-x**2-ratio*y**2)*x
   argDenom=math.fabs(p**2+x**2+y**2-2.*p*y)
   if argDenom < 1.e-40:
      intgrnd=0.
   else:
      intgrnd=nom/np.sqrt(argDenom)**3
   print '           vTran=%e, Vlong=%e, intgrndLong=%e' % (x,y,intgrnd)
   return intgrnd

def intgrndTran(x,y,ratio,p):
   nom=(p-x)*np.exp(-x**2-ratio*y**2)*x
#   denom=(p-x)**2+y**2
   denom=x**2+y**2
   if denom < 1.e-40:
      intgrnd=0.
   else:
      intgrnd=nom/np.sqrt(denom)**3
   print '           vTran=%e, Vlong=%e, intgrndTran=%e' % (x,y,intgrnd)
   return intgrnd

def intgrndTranS(x,ratio,p):
   nom=(p-x)*np.exp(-x**2)*x
   denom=(p-x)**2+1./ratio
   if denom < 1.e-40:
      intgrnd=0.
   else:
      intgrnd=nom/np.sqrt(denom)**3
   print '           vTran=%e, intgrndTran=%e' % (x,intgrnd)
   return intgrnd

def intLong(ratio,p,maxVeTran,maxVeLong):
   intgrl0=dblquad(lambda y,x: intgrndLong(x,y,ratio,p), 0.1, maxVeTran, \
                   lambda y: -maxVeLong, lambda y: maxVeLong,epsabs=1.e-15,epsrel=1.e-8)[0]
   intgrl=2./np.sqrt(pi)*np.sqrt(ratio)*intgrl0
#    print '    Long: intgrl0=%e, intgrl=%e' % (intgrl0,intgrl)   
   return intgrl

def intTran(ratio,p,maxVeTran,maxVeLong):
   intgrl0=dblquad(lambda y,x: intgrndTran(x,y,ratio,p), 0.1, maxVeTran, \
                   lambda y: -maxVeLong, lambda y: maxVeLong,epsabs=1.e-15,epsrel=1.e-8)[0]
   intgrl=2./np.sqrt(pi)*np.sqrt(ratio)*intgrl0
#    print '    Tran: intgrl0=%e, intgrl=%e' % (intgrl0,intgrl)   
   return intgrl

def intTranS(ratio,p,maxVeTran):
   intgrl0=quad(lambda x: intgrndTranS(x,ratio,p), 0., maxVeTran,epsabs=1.e-15,epsrel=1.e-8)[0]
   intgrl=2./np.sqrt(pi)*np.sqrt(ratio)*intgrl0
#    print '    Tran: intgrl0=%e, intgrl=%e' % (intgrl0,intgrl)   
   return intgrl

totalCPU=0 
totalELPSD=0

for i in range(nRelVtot):
   timeStart=os.times()
   relVtot[i]=minRelVtot+(maxRelVtot-minRelVtot)/nRelVtot*i
#    fricLong[i]=intLong(ratioTtran_Tlong,relVtot[i],maxVeTran,maxVeLong) 
#    fricTran[i]=intTran(ratioTtran_Tlong,relVtot[i],maxVeTran,maxVeLong) 
   fricTran[i]=intTranS(ratioTtran_Tlong,relVtot[i],maxVeTran) 
   timeEnd=os.times()
   deltaCPU=float(timeEnd[0])-float(timeStart[0])     # CPU time
   deltaELPSD=float(timeEnd[4])-float(timeStart[4])   # elapsed real time
   totalCPU +=deltaCPU
   totalELPSD +=deltaELPSD
   print 'i=%d, Vtot=%e, Long=%e, Tran=%e, CPU=%e, ELPSD=%e' % \
         (i,relVtot[i],fricLong[i],fricTran[i],deltaCPU,deltaELPSD)

print '\ntotalCPU=%e, totalELPSD=%e' % (totalCPU, totalELPSD)  

fig500=plt.figure(500)
# plt.semilogx(relVtot,fricLong,'-r',linewidth=2)   
line1, = plt.semilogx(relVtot,fricLong,'-xr',linewidth=2,label='Longitudinal')   
line2, = plt.semilogx(relVtot,fricTran,'-xb',linewidth=2,label='Transversal')   
plt.title('Friction Force', color='m',fontsize=20)
plt.xlabel('$V_p/v_{e\perp}$',color='m',fontsize=16)
plt.ylabel('Friction Force, rel.units',color='m',fontsize=16)
plt.xlim([minRelVtot,maxRelVtot])
plt.legend(handler_map={line1: HandlerLine2D()},loc='upper right')
plt.grid(True)

plt.show()   

sys.exit()   
