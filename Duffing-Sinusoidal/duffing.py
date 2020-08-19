# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:44:49 2020

@author: Amrita Punnavajhala
"""

import numpy as np
from numpy.random import random,rand
from scipy.integrate import odeint
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import plotly.offline as py
#import plotly.graph_objs as go
#from plotly.subplots import make_subplots
import matplotlib.animation as animation


seeds = 1


a0 = 1
a1 = 1

alpha = 1
delta = 0.25 # damping
gamma = 0.3 # forcing
beta = -1 
w = 1

time = 100
delTime = 0.001





interval=[-1,1]
X1=np.random.random(seeds)*(interval[1]-interval[0])+interval[0]
Y1=np.random.random(seeds)*(interval[1]-interval[0])+interval[0]

X2=np.random.random(seeds)*(interval[1]-interval[0])+interval[0]
Y2=np.random.random(seeds)*(interval[1]-interval[0])+interval[0]


def f(state,t):
    
    x1 = state[0]
    y1 = state[1]
    
    x2 = state[2]
    y2 = state[3]
    
    x1d = y1
    y1d = gamma*np.cos(w*t) - delta*y1 - beta*x1 - alpha*x1**3
    
    x2d = y2 + a1*w*x1*np.cos(w*t)+ y1*(a0+a1*np.sin(w*t)) -y2 -(x2-x1*(a0+a1*np.sin(w*t)))
    y2d = gamma*np.cos(w*t) - delta*y2 - beta*x2 - alpha*x2**3 + a1*w*y1*np.cos(w*t) + alpha*x2**3 + beta*x2 + delta*y2 - gamma*np.cos(w*t)+(a0+a1*np.sin(w*t))*(gamma*np.cos(w*t)-alpha*x1**3-beta*x1-delta*y1) -(y2-y1*(a0+a1*np.sin(w*t)))
    
    return[x1d,y1d,x2d,y2d]

t = np.arange(0.0, time, delTime)

for x1,y1,x2,y2 in zip(X1,Y1,X2,Y2):
    state0 = [x1,y1,x2,y2]
    
    state = odeint(f,state0,t, rtol=1.49012e-10, atol=1.49012e-10)
    
    
plt.plot(state[:,0],state[:,1])
plt.plot(state[:,2],state[:,3])
plt.figure()
plt.plot(t,state[:,2]/state[:,0])
plt.plot(t,state[:,3]/state[:,1])

