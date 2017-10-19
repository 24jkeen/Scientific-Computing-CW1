import scipy.integrate 
import numpy as np
import matplotlib.pyplot as plt
from math import *

def Duffing(U, t):
    
    return [U[1], -2*k*U[1] - U[0] + g*sin(w*t) - np.power(U[0],3)]


def Root(ys, x0):

    x0 = int(abs( x0 / end)* steps)
    ig = ys[x0]
    range_ = 1000
    time = ((x0 + range_)*end) / steps
    e = 0.001
    happy = False
    while happy == False:

        for i in range(range_):
            sg = ys[x0 + i + 10]

            if x0 + i + 11 == steps:
                break
            
            if abs(ig - sg) < e:
                happy = True
                time = ((x0 + i) * end) / steps
                break
        
        break

    return [happy, time] #round(time, 7)]

k = 0.05
g = 0.2
w = 1

start = 0
end = 200
steps = 2000


U0 = [0, 0]
ts = np. linspace(start, end, steps)
Us = scipy.integrate.odeint(Duffing, U0, ts)
ys = Us[:,0]



print(Root(ys, 140))






plt.xlabel("t")
plt.ylabel("u")
plt.title("Damped harmonic oscillator")
plt.plot(ts,ys);
plt.plot(ts,Us[:,1]);
plt.show()
