import numpy
from numpy import *
from sympy import *
from numpy.random import random,rand
from scipy.integrate import odeint
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.animation as animation
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

def VectorDivergence(vector,variables):
    D = []
    for vec in vector:
        row = []
        for var in variables:
            row.append(diff(vec,var))
        D.append(row)
    return Matrix(D)

#def renderAB(systemA,systemB,couplingA,couplingB,stabilizerA,stabilizerB,stateA0,stateB0,time=5,delTime=.1,plot=True):
#    """
#        rendering dynamics of the system for publication usage, using matplot
#    """

def syncError(truth,slave):
    error = truth - slave
    error = error**2
    return error

def dropna(array):
    array = array[~isnan(array)]
    return array

def lagMinimaSyncError(truth,slave,lags=10):
    lags = numpy.arange(-lags,lags,1,int)
    error = []
    for index,lag in enumerate(lags):
        err = syncError(truth,slave.shift(lag))
        err = dropna(err.sum(axis=1)).mean()
        error.append(err)
    index = argmin(error)
    min_err = error[index]
    min_lag = lags[index]
    return min_err,min_lag

def null(A,B,t):
    return 0

def evolveAB(systemA,systemB,stateA0,stateB0,couplingA=null,couplingB=null,stabilizerA=null,stabilizerB=null,time=5,delTime=None,tail=0,plot=True):
    """
        displaying dynamics of the system for notebook usage, using plotly
    """
    if delTime is None:
        timeline = time
    else:
        timeline = numpy.arange(0.0, time, delTime)
    dimA,dimB = len(stateA0),len(stateB0)
    def interaction(state,t):
        stateA,stateB = state[:dimA],state[dimA:]
        flowA = systemA(stateA,t) + couplingA(stateA,stateB,t) + stabilizerA(stateA,stateB,t)
        flowB = systemB(stateB,t) + couplingB(stateA,stateB,t) + stabilizerB(stateA,stateB,t)
        flow = numpy.concatenate((flowA,flowB))
        return flow
    state0 = numpy.concatenate((stateA0,stateB0))
    state = odeint(interaction, state0, timeline, rtol=1.49012e-10, atol=1.49012e-10)
    state = state[-tail:]
    stateA,stateB = state[:,:dimA],state[:,dimA:]
    if plot:
        traceA = tracingDynamics(stateA)
        traceB = tracingDynamics(stateB)
        py.iplot([traceA,traceB])
    return stateA,stateB

def tracingDynamics(state):
    dim = state.shape[1]
    trace = None
    if dim == 2:
        trace = go.Scatter(x=state[:,0],y=state[:,1],mode='lines')
    elif dim == 3:
        trace = go.Scatter3d(x=state[:,0],y=state[:,1],z=state[:,2],mode='lines')
    return trace

def evolveSystem3(system,state0,time=5,delTime=.1,plot=True):
    timeline = numpy.arange(0.0, time, delTime)
    state = odeint(system, state0, timeline, rtol=1.49012e-10, atol=1.49012e-10)
    if plot:
        trace = go.Scatter3d(x=state[:,0],y=state[:,1],z=state[:,2],mode='lines')
        py.iplot([trace])
    return state

def evolveSystem2(system,state0,time=5,delTime=.1,plot=True):
    timeline = numpy.arange(0.0, time, delTime)
    state = odeint(system, state0, timeline, rtol=1.49012e-10, atol=1.49012e-10)
    if plot:
        trace = go.Scatter(x=state[:,0],y=state[:,1],mode='lines')
        py.iplot([trace])
    return state
