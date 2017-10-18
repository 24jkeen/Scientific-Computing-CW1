import scipy.integrate 
import numpy as np
import matplotlib.pyplot as plt
from math import *

def Duffing(U, t):


    #u2 + 2*ku1 + u + u_3 = g*sin(wt)



    #0.5 <= w <= 1.5
    
    return [U[1], -2*k*U[1] - U[0] + g*sin(w*t) - np.power(U[0],3)]


k = 0.05
g = 0.2
w = 1.5


U0 = [0, 0]
ts = np. linspace(1000, 1050 , 200)
Us = scipy.integrate.odeint(Duffing, U0, ts)
ys = Us[:,0]

plt.xlabel("x")
plt.ylabel("y")
plt.title("Damped harmonic oscillator")
plt.plot(ts,ys);
plt.plot(ts,Us[:,1]);
plt.show()
