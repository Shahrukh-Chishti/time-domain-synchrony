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

timeline = np.arange(0.0, time, delTime)
def evolveSystem(system,N,state0,plot):
    state = odeint(system, state0, timeline, rtol=1.49012e-10, atol=1.49012e-10)