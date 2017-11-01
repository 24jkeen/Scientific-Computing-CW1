####### Import useful things ############

import scipy.integrate 
import matplotlib.pyplot as plt
#from math import *
from scipy.optimize import fsolve
from numpy import *
from numpy.matlib import repmat
from numpy import transpose


######## Define function to integrate in first order form ##########

def Duffing(U, t, pars):
    
    return [U[1], -2*pars[0]*U[1] - U[0] + pars[1]*sin(pars[2]*t) - power(U[0],3)]

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

########## Chebyshev section #############
## CHEB  compute D = differentiation matrix, x = Chebyshev grid

def cheb(N):

    """
    Generates the chebychev polynomials

    input:          Power of the polynomial you wish to use to approximate the integral
    
    
    output:         NxN matrix of Chebychev polynomials         
    """
    if N == 0:
        D = 0
        x = 0
        return (D, x)                                                           ##To catch any divisions by zero
    x = array([])                                                               ##Initialise empty arrays to fill later 
    c = array(2)

    for i in range(N+1):
        if i <= N:                                                              ## Get the x_i values, stopping at N
            x = append(x, cos( (pi * i ) / N  ))    
        if i == 0:
            pass                                                                ## C starts and ends with a 2 so we skip the first index of the vector
        if i == N-1:
            
            c = append(c, 2*(-1)**(i-1))                                        ## append the final 2
        if i > N-2 :                                                            ## Dont double count the last entry
            pass
            
        else:
            c = append(c, (-1)**(i-1))                                          ##fill the rest with +/- 1

    X = repmat(x,1, N+1)
    X = X.reshape(N +1, N +1)                                                   ##duplicate the column then reshape it into a matrix
    X = X.transpose() 
    dX = subtract(X , transpose(X))                                             ## construct d
    
    D = divide(tensordot( c, (1/c).transpose(), axes = 0), (dX + eye(N+1)))     ##Needs to be tensor product to output a matrix
   
    D + D.reshape(N+1,N+1)
    D = D - diag(sum(D, axis = 1  ))                                            ## subtracts a diagonal matrix of the sum of each row
    return (D, x)


# Tests on the Chebyshev differentiation matrix routines

 
def testeq(M, N, name):
    """
    testeq(M, N, name)

    Test for equality of two matrices (M and N) allowing for some floating point
    error. The parameter name is the name of the test.
    """
    D = abs(M - N)
    if all(D < 0.01):
        print("Passed " + name)
    else:
        print("Failed " + name)



def testp(M,  name):
    """
    
    """
    if all(M[len(M)-1] - M[0] < 0.001):
        print("Failed " + name)

    else:
        print("Passed " + name)

    


def runtests_cheb(cheb):
    """
    runtests_cheb(cheb)

    Run a small suite of tests on the Chebyshev differentiation matrix code
    provided. E.g.,

        from week5_chebtests import runtests_cheb
        runtests_cheb(mychebcode)
    """
    # Test 1
    M = cheb(1)
    N = array([[0.5, -0.5], [0.5, -0.5]])
    testeq(M, N, "test 1")

    # Test 2
    M = cheb(2)
    N = array([[1.5, -2.0, 0.5], [0.5, 0.0, -0.5], [-0.5, 2.0, -1.5]])
    testeq(M, N, "test 2")

    # Test 3
    M = cheb(5)
    N = array([[8.5,-10.472136,2.8944272,-1.527864,1.1055728,-0.5],
               [2.618034,-1.1708204,-2,0.89442719,-0.61803399,0.2763932],
               [-0.7236068,2,-0.17082039,-1.618034,0.89442719,-0.38196601],
               [0.38196601,-0.89442719,1.618034,0.17082039,-2,0.7236068],
               [-0.2763932,0.61803399,-0.89442719,2,1.1708204,-2.618034],
               [0.5,-1.1055728,1.527864,-2.8944272,10.472136,-8.5]])
    testeq(M, N, "test 3")


def difftests(cheb):
    """
    This is a suite of tests that checks to see if the Chebyshev method is working using some known derivatives

    input:      difftests(cheb)
    output:     passed test1, passed test2, failed test3 etc

    """

    # Test 1
    D, x = cheb(4)
    Q = dot(D , cos(x))
    A = -sin(x)
    testeq(Q, A, 'test1')
    testp(Q, "Periodic Test1")

    #Test 2
    D, x = cheb(10)
    Q = dot(D , exp(x))
    A = exp(x)
    testeq(Q, A, 'test2')
    testp(Q, 'Periodic Test2')


    #Test 3
    D, x = cheb(5)
    Q = dot(D , x**2)
    A = 2 * x
    testeq(Q, A, 'test3')
    testp(Q, 'Periodic Test3')


















######### System constants ##########
k = 1
g = 0.2
w = 1.2
pars = [k, g, w]

####### Timeings of the integrator ##########
start = 0
end = 200
steps = 20000

######## Integrating ###########
U0 = [0, 1]                                                 ## Initial values
ts = linspace(start, end, steps)
Us = scipy.integrate.odeint(Duffing, U0, ts, args=(pars,))
ys = Us[:,0]                                                ## Extracts the y axis values


######## Shooting method ##########
U0 = shooting(Duffing, U0, 2*pi/w, pars)

######## Solve for one periodic orbit ########
t = linspace(0, 2*pi / w , 500)
x = scipy.integrate.odeint(Duffing, U0, t, args= (pars,))

####### Dastardly Plotting ############

#plt.plot(t, x);                                            ## plots the solution to the first first order eqn
#plt.plot(ys,Us[:,1]);                                       ## '' but for the second
#plt.show()


def Results( ODE, U0, pars, vary_par, step_size, max_steps, discretisation, solver):
    incorrect_input = False
    max_x_values = []
    max_x_cheb_values = []
    par_values = [] 
    U0 = shooting(ODE, U0, 2*pi/w, pars)
    t = linspace(0, 2 * pi / w, 501)

    for i in range(max_steps):
        
        pars[vary_par] += step_size
        par_values.append( pars[vary_par] )
        

        if discretisation == 'odeint':    
            x = scipy.integrate.odeint(ODE, U0, t, args = (pars,))
            max_x = max(x[1, :])
            max_x_values.append(max_x)
            plt.plot(par_values, max_x_values)
            plt.xlabel("Varied Parameter")
            plt.ylabel("x Max")
            plt.title(" Solving BVP's with odeint ")



        elif discretisation == 'chebyshev':
            x_cheb = run_cheb(ODE, 101, U0, (pars,))    
            max_x_cheb = max(x_cheb[1,:] )
            max_x_cheb_values.append(max_x_cheb)
            plt.plot(par_values, max_x_cheb_values)
            plt.xlabel("Varied Parameter")
            plt.ylabel("x Max")
            plt.title("Solving BVP's with Chebyshev approximations")
        
        else:
            incorrect_input = True
            
            
    if incorrect_input == True:
        print("Please enter a discretisation: \n 'odeint' \n 'chebyshev")
        
    plt.show()




def run_cheb(f, N, U0, args):
    D, x = cheb(N)
    A_0 = D * f(U0, x, pars)[1]
    return A_0

Results(Duffing, [0, 1], pars, 1, 0.01, 2000, 'odeint', 0) 


