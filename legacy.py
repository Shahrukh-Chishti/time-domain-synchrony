# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:59:03 2019

@author: Amrita Punnavajhala
"""

import numpy
import numpy as np
from numpy.random import random,rand
from scipy.integrate import odeint
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.animation as animation
from plotly.subplots import make_subplots

seeds=1
sigma = 10
rho = 28
beta = 8/3

time = 50
delTime = 0.001

eps = 1

a0=1
a1=100
w=1.0

l=int(time/delTime)

interval=[-10,10]
X1=np.random.random(seeds)*(interval[1]-interval[0])+interval[0]
Y1=np.random.random(seeds)*(interval[1]-interval[0])+interval[0]
Z1=np.random.random(seeds)*(interval[1]-interval[0])+interval[0]

X2=np.random.random(seeds)*(interval[1]-interval[0])+interval[0]
Y2=np.random.random(seeds)*(interval[1]-interval[0])+interval[0]
Z2=np.random.random(seeds)*(interval[1]-interval[0])+interval[0]

def phi1(x1,y1,z1,x2,y2,z2,t):
    return a1*w*x1*np.cos(t*w)-sigma*(-a0-a1*np.sin(t*w))*(-x1+y1)-sigma*(-x1*(a0+a1*np.sin(t*w))+y1*(a0+a1*np.sin(t*w)))-(x2-x1*(a0+a1*np.sin(w*t)))

def phi2(x1,y1,z1,x2,y2,z2,t):
    return a1*w*y1*np.cos(t*w)-x1*(a0+a1*np.sin(t*w))*(rho-z1*(a0+a1*np.sin(t*w)))+y1*(a0+a1*np.sin(t*w))-(-a0-a1*np.sin(t*w))*(x1*(rho-z1)-y1)-(y2-y1*(a0+a1*np.sin(w*t)))

def phi3(x1,y1,z1,x2,y2,z2,t):
    return a1*w*z1*np.cos(t*w)+beta*z1*(a0+a1*np.sin(t*w))-x1*y1*((a0+a1*np.sin(t*w))**2)-(-a0-a1*np.sin(t*w))*(-beta*z1+x1*y1)-(z2-z1*(a0+a1*np.sin(w*t)))

def coupledLorenz(state,t):
    x1 = state[0]
    y1 = state[1]
    z1 = state[2]

    x2 = state[3]
    y2 = state[4]
    z2 = state[5]

    x1d = sigma*(y1-x1)
    y1d = x1*(rho-z1)-y1
    z1d = x1*y1-beta*z1

    x2d = sigma*(y2-x2) + phi1(x1,y1,z1,x2,y2,z2,t)
    y2d = x2*(rho-z2)-y2 + phi2(x1,y1,z1,x2,y2,z2,t)
    z2d = x2*y2-beta*z2 + phi3(x1,y1,z1,x2,y2,z2,t)

    return [x1d, y1d, z1d, x2d, y2d, z2d]

t = np.arange(0.0, time, delTime)
i=1

for x1,y1,z1,x2,y2,z2 in zip(X1,Y1,Z1,X2,Y2,Z2):
    state0 = [x1,y1,z1,x2,y2,z2]
    print (state0)
    state = odeint(coupledLorenz, state0, t, rtol=1.49012e-10, atol=1.49012e-10)
    print ('******System ',i,'*******')
    i+=1
    print ('Terminal State 1:',state[-1,:3])
    print ('Terminal State 2:',state[-1,3:])

    L1 = state[:,:3]
    L2 = state[:,3:]
    plt.figure()
    plt.grid(True)
    #plt.legend()
    #plt.show()
    #StabilityPlot(state)
    trace1=go.Scatter3d(x=state[:,0],y=state[:,1],z=state[:,2],mode='lines')
    trace2=go.Scatter3d(x=state[:,3],y=state[:,4],z=state[:,5],mode='lines')
    data=[]
    data.append(trace1)
    data.append(trace2)
    #py.plot(data)

constraint=np.zeros(l)
for i in range(0,len(t)):
    constraint[i]=a0+a1*np.sin(w*t[i])

transient_time = 0
fig = make_subplots(shared_xaxes=True,rows=2, cols=1)
# Constraints
#plt.plot(constraint[transient_time:])
traceConstraint = go.Scatter(name='constraint',x=t[transient_time:],y=constraint[transient_time:])


# Order Parameter Ratio
#plt.plot(L2[transient_time:,0]/L1[transient_time:,0],label="$x_2/x_1$")
#plt.plot(L2[transient_time:,1]/L1[transient_time:,1],label="$y_2/y_1$")
#plt.plot(L2[transient_time:,2]/L1[transient_time:,2],label="$z_2/z_1$")
traceOP_x = go.Scatter(name='order Parameter X',x=t[transient_time:],
                      y=L2[transient_time:,0]/L1[transient_time:,0])
traceOP_y = go.Scatter(name='order Parameter y',x=t[transient_time:],
                      y=L2[transient_time:,1]/L1[transient_time:,1])
traceOP_z = go.Scatter(name='order Parameter z',x=t[transient_time:],
                      y=L2[transient_time:,2]/L1[transient_time:,2])

# Eigenvalues

evI = np.zeros(len(t))
evII = numpy.zeros(len(t))
for index,i in enumerate(t):
    evI[index] = -((a0+a1*np.sin(w*i))*(a0+a1*w*np.cos(w*i)+a1*np.sin(w*i)))-1
    evII[index] = -(a0+a1*np.sin(w*i))**2 - 1

traceEVI = go.Scatter(name='eigI',x=t[transient_time:],y=evI[transient_time:])
traceEVII = go.Scatter(name='eigII',x=t[transient_time:],y=evII[transient_time:])

fig.add_trace(traceConstraint,row=1,col=1)
fig.add_trace(traceOP_x,row=2,col=1)
fig.add_trace(traceOP_y,row=2,col=1)
fig.add_trace(traceOP_z,row=2,col=1)

fig.add_trace(traceEVI,row=3,col=1)
fig.add_trace(traceEVII,row=3,col=1)

fig.show()

sys.exit()

ax=fig.add_axes([0,0,1,1],projection='3d')
ax.plot(state[:,0],state[:,1],state[:,2])
ax.plot(state[:,3],state[:,4],state[:,5])
ax.grid(False)

############################################# EIGENVALUES ####################################################

