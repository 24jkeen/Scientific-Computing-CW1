####### Import useful things ############

import scipy.integrate 
import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.optimize import fsolve


######## Define function to integrate in first order form ##########

def Duffing(U, t, pars):
    
    return [U[1], -2*pars[0]*U[1] - U[0] + pars[1]*sin(pars[2]*t) - np.power(U[0],3)]

######## Define shooting method function ###########
def Root(ys, x0):

    x0 = int(abs( x0 / end)* steps)                         ## The input is a normal time
    ig = ys[x0]                                             ## we need to convert this into an index for the list of ys    
    time = ((x0 + steps)*end) / steps
    e = 0.001                                               ## this constant is the error that we will allow in finding
    padding = int(steps / 200)                                                        ##a value since we do not have a continuous function
    happy = False
    while happy == False:                                   ##while we havent found the answer ... 

        for i in range(steps):
            sg = ys[x0 + i + int(((w - end) / end * steps)) + padding]  ## so the function doesnt find 'itself' because it could be flat at ig

            if x0 + i + 11 == steps:                        ## avoids indexing errors
                break
            
            if abs(ig - sg) < e:                            ##if initial guess and second guess are within the limit
                happy = True
                time = ((x0 + i) * end) / steps             ## tells us when the second 'root' appears 
                break
        
        break

    return [happy, time]

######### More modely solutions #############
def zeroproblem(x, ODE, T, pars):
    return x - scipy.integrate.odeint(ODE, x, [0, T], args = (pars,))[-1, :]


def shooting(ODE, x0, T, pars):
    xnew, info, ier, mesg = fsolve(zeroproblem, x0, args = (ODE, T, pars), full_output = True)
    if ier == 1:
        return xnew
    else:
        return nan

######### System constants ##########
k = 0.05
g = 0.2
w = 1.2
pars = [k, g, w]

####### Timeings of the integrator ##########
start = 0
end = 200
steps = 20000

######## Integrating ###########
U0 = [0, 1]                                                 ## Initial values
ts = np. linspace(start, end, steps)
Us = scipy.integrate.odeint(Duffing, U0, ts, args=(pars,))
ys = Us[:,0]                                                ## Extracts the y axis values


######## Shooting method ##########
U0 = shooting(Duffing, U0, 2*pi/w, pars)

######## Solve for one periodic orbit ########
t = np.linspace(0, 2*pi / w , 500)
x = scipy.integrate.odeint(Duffing, U0, t, args= (pars,))

####### Dastardly Plotting ############
plt.xlabel("t")
plt.ylabel("u")
plt.title("Damped harmonic oscillator")
plt.plot(t, x);                                            ## plots the solution to the first first order eqn
#plt.plot(ys,Us[:,1]);                                       ## '' but for the second
plt.show()
