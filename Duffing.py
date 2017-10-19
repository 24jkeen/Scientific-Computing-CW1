import scipy.integrate 
import numpy as np
import matplotlib.pyplot as plt
from math import *

def Duffing(U, t):
    
    return [U[1], -2*k*U[1] - U[0] + g*sin(w*t) - np.power(U[0],3)]


def Root(ys, x0):

    x0 = int(abs( x0 / end)* steps)
    ig = ys[x0]
    time = 0
    range_ = 1000
    e = 0.00000000000000000001
    happy = False
    while ~happy:

        for i in range(range_):
            sg = ys[x0 + i + 1]

            if x0 + i + 2 == steps:
                break
            
            elif abs(ig - sg) < e:
                happy = True
                time = ((x0 + i) * end) / steps
                break
        break

    return [happy, round(time, 3)]

k = 0.05
g = 0.2
w = 1

start = 0
end = 10
steps = 20000


U0 = [0, 0]
ts = np. linspace(start, end, steps)
Us = scipy.integrate.odeint(Duffing, U0, ts)
ys = Us[:,0]



print(Root(ys, 0))






plt.xlabel("t")
plt.ylabel("u")
plt.title("Damped harmonic oscillator")
plt.plot(ts,ys);
plt.plot(ts,Us[:,1]);
plt.show()
