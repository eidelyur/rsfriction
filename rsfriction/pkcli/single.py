from __future__ import print_function, division
import numpy as np
import scipy as sp
from mpmath import mp
from scipy import constants as const
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# set decimal precision
# binary prec ~ 3.33*dps
mp.dps = 80
print(mp)


class Wrapper(object):
    
    def __init__(self, value=None):
        # slow for clarity - improve later
        if value is not None:
            self._value = value
        else:
            self._value = np.zeros(shape=(3,4))
    
    @property
    def x_i(self):
        return self._value[0][0]
        
    @x_i.setter
    def x_i(self, v):
        self._value[0][0] = v
    
    @property
    def y_i(self):
        return self._value[0][1]
        
    @y_i.setter
    def y_i(self, v):
        self._value[0][1] = v
    
    @property
    def z_i(self):
        return self._value[0][2]
    
    @z_i.setter
    def z_i(self, v):
        self._value[0][2] = v
    
    #
    
    @property
    def p_ix(self):
        return self._value[1][0]
        
    @p_ix.setter
    def p_ix(self, v):
        self._value[1][0] = v
    
    @property
    def p_iy(self):
        return self._value[1][1]
        
    @p_iy.setter
    def p_iy(self, v):
        self._value[1][1] = v
    
    @property
    def p_iz(self):
        return self._value[1][2]
        
    @p_iz.setter
    def p_iz(self, v):
        self._value[1][2] = v
    
    #
    
    @property
    def x_e(self):
        return self._value[2][0]
        
    @x_e.setter
    def x_e(self, v):
        self._value[2][0] = v
    
    @property
    def y_e(self):
        return self._value[2][1]
        
    @y_e.setter
    def y_e(self, v):
        self._value[2][1] = v
    
    @property
    def z_e(self):
        return self._value[2][2]
    
    @z_e.setter
    def z_e(self, v):
        self._value[2][2] = v
    
    #
    
    @property
    def p_ex(self):
        return self._value[3][0]
        
    @p_ex.setter
    def p_ex(self, v):
        self._value[3][0] = v
    
    @property
    def p_ey(self):
        return self._value[3][1]
        
    @p_ey.setter
    def p_ey(self, v):
        self._value[3][1] = v
    
    @property
    def p_ez(self):
        return self._value[3][2]
        
    @p_ez.setter
    def p_ez(self, v):
        self._value[3][2] = v


def M0_dt(w, dt):
    
    # ion mass = proton mass
    m_i = const.m_p
    
    magnetic_field = 1. # Tesla, will be user defined
    
    # gryrotron frequency
    omega_e = np.abs(const.e * magnetic_field) / const.m_e
    
    V_ex = w.p_ex / const.m_e + omega_e * w.y_e
    V_ey = w.p_ey / const.m_e
    
    # phase angle
    phi_0 = np.arctan(V_ex / V_ey)
    
    # perpendicular velocity (V_perp)
    gc_velocity = np.sqrt(V_ex**2 + V_ey**2)
    
    p_gc = gc_velocity / omega_e
    
    xbar_e = w.x_e + p_gc * (np.cos(phi_0 + omega_e * dt) - np.cos(phi_0))
    ybar_e = w.y_e - p_gc * (np.sin(phi_0 + omega_e * dt) - np.sin(phi_0))
    zbar_e = w.z_e + (w.p_ez / const.m_e) * dt
    
    xbar_i = w.x_i + (w.p_ix / m_i) * dt
    ybar_i = w.y_i + (w.p_iy / m_i) * dt
    zbar_i = w.z_i + (w.p_iz / m_i) * dt
    
    return mp.matrix(
        ((xbar_e, ybar_e, zbar_e),
        (xbar_i, ybar_i, zbar_i))
    )


def Mc_dt(w, dt):
    N = 1 # user input
    # ion charge 
    Q_i = const.elementary_charge * N
    b = np.sqrt((w.x_i - w.x_e)**2 + (w.y_i - w.y_e)**2 + (w.z_i - w.z_e)**2)
    print('b:', b)
    
    alpha = ((const.e * Q_i) / (4. * const.pi * const.epsilon_0)) * dt
    print('a:', alpha)
    
    pbar_ix = mp.mpf(w.p_ix) - mp.mpf((alpha * (w.x_i - w.x_e) / b**3))
    pbar_iy = mp.mpf(w.p_iy) - mp.mpf((alpha * (w.y_i - w.y_e) / b**3))
    pbar_iz = mp.mpf(w.p_iz) - mp.mpf((alpha * (w.z_i - w.z_e) / b**3))
    
    pbar_ex = -(pbar_ix - w.p_ix) + w.p_ex
    pbar_ey = -(pbar_iy - w.p_iy) + w.p_ey
    pbar_ez = -(pbar_iz - w.p_iz) + w.p_ez
    
    return mp.matrix(
        ((pbar_ix, pbar_iy, pbar_iz),
        (pbar_ex, pbar_ey, pbar_ez))
    )


def init(w):
    delt = const.value('classical electron radius')
    w.x_i = w.x_i + mp.mpf('3.23e3') * delt
    w.y_i = w.y_i - mp.mpf('1.67e3') * delt
    w.z_i = w.z_i + mp.mpf('2.52e3') * delt


def step(w,dt):
    return (M0_dt(w, dt/2), Mc_dt(w, dt))


if __name__ == '__main__':
    magnetic_field = 1. # Tesla, will be user defined
    # Gyrotron Frequency
    omega_e = np.abs(const.e * magnetic_field) / const.m_e
    dt_max = 1/8 * 2*const.pi / omega_e
    # 
    for N in [1,2,4,8]:
        v = np.ones(shape=(4,3), dtype=np.float_)
        w = Wrapper(v)    
        init(w)
        dt = 1./N * dt_max
        print(step(w, dt))

