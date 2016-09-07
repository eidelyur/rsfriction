from __future__ import print_function, division
from collections import namedtuple
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


Index = namedtuple('Index',[
    'x_i',
    'y_i',
    'z_i',
    'p_ix',
    'p_iy',
    'p_iz',
    'x_e',
    'y_e',
    'z_e',
    'p_ex',
    'p_ey',
    'p_ez',
])

idx = Index(*range(12))

def M0_dt(w, dt):
    
    # ion mass = proton mass
    m_i = const.m_p
    
    magnetic_field = 1. # Tesla, will be user defined
    
    # gryrotron frequency
    omega_e = np.abs(const.e * magnetic_field) / const.m_e
    
    V_ex = w[idx.p_ex] / const.m_e + omega_e * w[idx.y_e]
    V_ey = w[idx.p_ey] / const.m_e
    
    # phase angle
    phi_0 = np.arctan(V_ex / V_ey)
    
    # perpendicular velocity (V_perp)
    gc_velocity = np.sqrt(V_ex**2 + V_ey**2)
    
    p_gc = gc_velocity / omega_e
    
    w[idx.x_e] = w[idx.x_e] + p_gc * (np.cos(phi_0 + omega_e * dt) - np.cos(phi_0))
    w[idx.y_e] = w[idx.y_e] - p_gc * (np.sin(phi_0 + omega_e * dt) - np.sin(phi_0))
    w[idx.z_e] = w[idx.z_e] + (w[idx.p_ez] / const.m_e) * dt
    
    w[idx.x_i] = w[idx.x_i] + (w[idx.p_ix] / m_i) * dt
    w[idx.y_i] = w[idx.y_i] + (w[idx.p_iy] / m_i) * dt
    w[idx.z_i] = w[idx.z_i] + (w[idx.p_iz] / m_i) * dt
    
    return w
        


def Mc_dt(w, dt):
    N = 1 # user input
    # ion charge 
    Q_i = const.elementary_charge * N
    b = np.sqrt((w[idx.x_i] - w[idx.x_e])**2 + (w[idx.y_i] - w[idx.y_e])**2 + (w[idx.z_i] - w[idx.z_e])**2)
    alpha = ((const.e * Q_i) / (4. * const.pi * const.epsilon_0)) * dt
    
    w[idx.p_ix] = mp.mpf(w[idx.p_ix]) - mp.mpf((alpha * (w[idx.x_i] - w[idx.x_e]) / b**3))
    w[idx.p_iy] = mp.mpf(w[idx.p_iy]) - mp.mpf((alpha * (w[idx.y_i] - w[idx.y_e]) / b**3))
    w[idx.p_iz] = mp.mpf(w[idx.p_iz]) - mp.mpf((alpha * (w[idx.z_i] - w[idx.z_e]) / b**3))
    
    w[idx.p_ex] = -(w[idx.p_ix] - w[idx.p_ix]) + w[idx.p_ex]
    w[idx.p_ey] = -(w[idx.p_iy] - w[idx.p_iy]) + w[idx.p_ey]
    w[idx.p_ez] = -(w[idx.p_iz] - w[idx.p_iz]) + w[idx.p_ez]
    
    return w


def initialize(w):
    delt = const.value('classical electron radius')
    w[idx.x_i] = w[idx.x_i] + mp.mpf('3.23e3') * delt
    w[idx.y_i] = w[idx.y_i] - mp.mpf('1.67e3') * delt
    w[idx.z_i] = w[idx.z_i] + mp.mpf('2.52e3') * delt


def ts_vec(dt, t_max):
    t = 0.
    while t <= t_max:
        yield t
        t = t + dt
    yield t


def step(w,dt):
    return (M0_dt(w, dt/2), Mc_dt(w, dt))


if __name__ == '__main__':
    magnetic_field = 1. # Tesla, will be user defined
    # Gyrotron Frequency
    omega_e = np.abs(const.e * magnetic_field) / const.m_e
    dt_max = 1/8 * 2*const.pi / omega_e
    #
    t_max = 0.000001
    #
    for N in [1,2,4,8][1:2]:
        w = np.ones(shape=(12), dtype=np.float_)
        initialize(w)
        print(mp.matrix(w))
        dt = 1./N * dt_max
        print(step(w, dt))
        ts = [_t for _t in ts_vec(dt, t_max)]
        r1 = odeint(M0_dt, w, ts)
        print(r1)
        r2 = odeint(Mc_dt, w, ts)
        print(r2)

