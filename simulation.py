import numpy
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

def evolveSystem3(system,state0,time=5,delTime=.1,plot=True):
    timeline = numpy.arange(0.0, time, delTime)
    state = odeint(system, state0, timeline, rtol=1.49012e-10, atol=1.49012e-10)
    if plot:
        trace = go.Scatter3d(x=state[:,0],y=state[:,1],z=state[:,2],mode='lines')
        py.iplot([trace])
        
def evolveSystem2(system,state0,time=5,delTime=.1,plot=True):
    timeline = numpy.arange(0.0, time, delTime)
    state = odeint(system, state0, timeline, rtol=1.49012e-10, atol=1.49012e-10)
    if plot:
        trace = go.Scatter(x=state[:,0],y=state[:,1],mode='lines')
        py.iplot([trace])