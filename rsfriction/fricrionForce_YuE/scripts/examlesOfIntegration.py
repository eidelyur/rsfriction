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


import scipy.integrate as integrate
from scipy.integrate import quad, nquad, dblquad

#
# 1D integrals (function integrate.quad)
#

#
#  1) int_0^inf exp(-x^2):
#
def integrand1(x):
   return np.exp(-x**2)
   
result1=2./np.sqrt(pi)*quad(integrand1,0,np.inf)[0]
print 'result1=%e' % result1

# result1=1.00000+00, i.e.OK!

result2=2./np.sqrt(pi)*quad(lambda x: integrand1(x), 0, np.inf)[0]
print 'result2=%e' % result2

# result2=1.00000+00, i.e.OK!

#
#  2) sqrt(pi)/2.*(int_0^inf exp(-x^2)*erf(x)):
#
def integrand3(x):
   return np.exp(-x**2)*math.erf(x)
   
result3=np.sqrt(pi)/2.*quad(integrand3,0,np.inf)[0]
print 'result3=%e' % result3

# result3=3.926991-01, i.e. OK (see example 2) for 2D integation)

#
# 1D integral with parameter (function quad)
#
#  3) sqrt(pi/a)/2.*(int_0^inf exp(-x^2)*erf(sqrt(a)*x)):
#     if a=1 example moves to examples 2) for 1D):
#

# Repetition without parameter, but using quad inside def (works!):
#
# def integrand3a(x):
#    return np.exp(-x**2)*math.erf(x)
# 
# def int_integrand3a():
#    return quad(integrand3a,0,np.inf)[0]
# 
# result3a=np.sqrt(pi)/2.*int_integrand3a()
# print 'result3a=%e' % result3a

# Now with parameter:
#
def integrand3a(x,a):
   return np.exp(-x**2)*math.erf(np.sqrt(a)*x)

def int_integrand3a(a):
   intgrl=quad(lambda x: integrand3a(x,a), 0, np.inf)[0]
   return np.sqrt(pi/a)/2.*intgrl

for p in (0.5,1.0,1.5):
   result3a=int_integrand3a(p)
   print 'result3a=%e' % result3a


#
# 2D integrals (function integrate.dblquad)
#

#
#  1) int_0^inf exp(-x^2-y^2):
#
def integrand4(x,y):
   return np.exp(-x**2-y**2)

result4=4./pi*dblquad(lambda x,y: integrand4(x,y), 0, np.inf, \
                      lambda x: 0, lambda x: np.inf)[0]
print 'result4=%e' % result4
   
#
#  2) int_x_0^inf int_y_0^x exp(-x^2-y^2)  
#     (= sqrt(pi)/2*(int_0^inf exp(-x^2)*erf(x))):
#
timeStart=os.times()
for i in range(10):
   result5=dblquad(lambda x,y: integrand4(x,y), 0, np.inf, \
                   lambda x: 0, lambda x: x)[0]
timeEnd=os.times()
print 'result5=%e' % result5
#
# result5=3.926991-01, i.e. OK (see example 2) for 1D integation)
#

# "time games": os.times() returns tuple with CPU time, system time,
# children's CPU time, children's system time, and elapsed time (from
# year's begin  ?); all times in seconds  

print 'timeStart=',timeStart
print 'timeEnd=',timeEnd
delta1=float(timeEnd[0])-float(timeStart[0])   # CPU time
delta2=float(timeEnd[4])-float(timeStart[4])   # elapsed real time
print 'delta1=%e, delta2=%e' % (delta1, delta2)  
#
#delta1=1.121000e+01, delta2=1.119000e+01 (for 1000 cycles)
#

#
# 2D integrals (function integrate.nquad)
#

#
#  3) int_x_0^inf int_y_0^x exp(-x^2-y^2)  
#     (= sqrt(pi)/2*(int_0^inf exp(-x^2)*erf(x))):
#
def integrand6(x,y):
   return np.exp(-x**2-y**2)

def bounds_x():
   return [0,np.inf]
   
def bounds_y(x):
   return [0,x]

timeStart=os.times()
for i in range(10):
   result6=nquad(integrand6,[bounds_y,bounds_x])[0]
timeEnd=os.times()
print 'result6=%e' % result6
#
# result6=3.926991-01, i.e. OK (see example 2) for 1D integation)
#
print 'timeStart=',timeStart
print 'timeEnd=',timeEnd
delta3=float(timeEnd[0])-float(timeStart[0])   # CPU time
delta4=float(timeEnd[4])-float(timeStart[4])   # elapsed real time
print 'delta3=%e, delta4=%e' % (delta3, delta4)  
#
#delta3=1.149000e+01, delta4=1.149000e+01 (for 1000 cycles)
#
#--------------------------------------------
#  So, nquad some slower than dblquad!!!
#--------------------------------------------

#
# 2D integrals with parameter (function integrate.dblquad)
#

#
#  1) int_x_0^inf int_y_0^x exp(-x^2-a*y^2)  
#     (= sqrt(pi/a)/2*(int_0^inf exp(-x^2)*erf(sqrt(a)*x));
#     if a=1 example moves to examples 2),3) for 2D):
#

sys.exit()   

