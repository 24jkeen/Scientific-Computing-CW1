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
    """
        The Duffing equation
        Takes the form $d^2x/dt^2 + zdx/dt + ax + Bx^3 = gcos(wt) $
        inputs      Initial conditions U;
                    time t;
                    Other parameters z, a, B, g, w

        outputs     2D matrix of y values for the duffing equation
    """
    return [U[1], -2*pars[0]*U[1] - U[0] + pars[1]*sin(pars[2]*t) - power(U[0],3)]

######### More modely solutions #############

def zeroproblem(x, ODE, T, pars):
    """
        Returns an ODE in the form x - dx/dt = e so that e can then be minimised to find a root to the equation
        inputs      x values
                    ODE in question
                    The time period to look for
                    extra parameters

        outputs     an ODE in a form that can be passed to a root finder
    """
    
    return x - scipy.integrate.odeint(ODE, x, [0, T], args = (pars,))[-1, :]


def shooting(ODE, x0, T, pars):
    """
        Finds the roots of an ODE
        inputs      ODE in question
                    An initial guess
                    The time period to look for
                    extra parameters for the ODE
       
        outputs     a root to the ode if one exists (if the initial guess is too far out of range will return NaN)
    """


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



# Tests on the collocation method routines


def ode1(x, t, par):
    """
    ode1(x, t, par)

    Return the right-hand-side of the ODE

        x' = sin(pi*t) - x
    """
    return sin(pi*t) - x     # [:, 0]


def ode2(x, t, par):
    """
    ode2(x, t, par)

    Return the right-hand-side of the ODE

        x'' + par[0]*x' + par[1]*x = sin(pi*t)
    """
    return [x[1], sin(pi*t) - par[0]*x[1] - par[1]*x[0]]


def runtests_ode(collocation):
    """
    runtests_ode(collocation)

    Run a small suite of tests on the collocation code provided. E.g.,

        from week5_odetests import runtests_ode
        runtests_ode(mycollocationcode)

    The collocation function should take the form

        collocation(ode, n, x0, pars)

    where ode is the right-hand-side of the ODE, n is the order of the
    polynomial to use, x0 is the initial guess at the solution, and pars is the
    parameters (if any, use [] for none).
    """
    # Solve the first ODE on the interval [-1, 1] with no parameters and zeros
    # as the starting guess; use 21 points across the interval
    n = 20  # 21 - 1 for the degree of polynomial needed
    x = cos(pi*arange(0, n+1)/n)  # the Chebyshev collocation points
    soln1 = collocation(ode1, n, zeros(2*(n+1)), [])
    plt.plot(x, soln1)
    plt.show()
    exactsoln1 = 1/(1+pi**2)*sin(pi*x) - pi/(1+pi**2)*cos(pi*x)
    if linalg.norm(soln1 - exactsoln1) < 1e-6:
        print("ODE test 1 passed")
    else:
        print("ODE test 1 failed")



def run_cheb(f, N, U0, pars):
    D, t = cheb(N)
    
    def coll(x, pars):
                
        p = zeros([2*(N+1)])
        h = dot(D, transpose(x)) - transpose(f(x, t, pars))
        g = x[0] - x[N]

        for i in range(len(2*x)):
            if i % 2 == 0:
                p[i] = h[int(i/2)]
            else:
                p[i] = g

        return p 


    def reshaped(x, pars):
        
        return reshape (coll( reshape( x, [2, (N + 1)]), pars ), [2*(N+1), 1]   )

    def new_f(x):

        for i in range(len(2*x)):
            if i % 2 == 0:
                pass #[i] = dot(D[i, :], x[i] - 
    
    A_0 = fsolve(coll, U0, pars)
    return A_0


runtests_ode(run_cheb)






def Results( ODE, U0, pars, vary_par, step_size, max_steps, discretisation, solver):
    incorrect_input = False
    max_x_values = []
    max_x_cheb_values = []
    par_values = [] 
    U0 = shooting(ODE, U0, 2*pi/w, pars)
    t = linspace(0, 2 * pi / w, 501)
    print(pars)
   
    if discretisation.upper() == 'ODEINT':
        for i in range(max_steps):
            pars[vary_par] += step_size
            par_values.append(pars[vary_par])
            print(i)
            U0 = shooting(ODE, U0, 2*pi/w, pars)
            x = scipy.integrate.odeint(ODE, U0, t, args = (pars,))
            max_x = max(x[1, :])
            max_x_values.append(max_x)
            plt.plot(par_values, max_x_values)
            plt.xlabel("Varied Parameter")
            plt.ylabel("x Max")
            plt.title(" Solving BVP's with odeint ")



    if discretisation.upper() == 'CHEBYSHEV':
        U0 = shooting(ODE, U0, 2*pi/w, pars)
        x_cheb = run_cheb(ODE, 101, U0, (pars,))    
        max_x_cheb = max(x_cheb[1,:] )
        max_x_cheb_values.append(max_x_cheb)
        plt.plot(par_values, max_x_cheb_values)
        plt.xlabel("Varied Parameter")
        plt.ylabel("x Max")
        plt.title("Solving BVP's with Chebyshev approximations")
    
    else:
        incorrect_input = True
        
            
    #if incorrect_input == True:
    #    print("Please enter a discretisation: \n 'odeint' \n 'chebyshev")
        
    plt.show()








######### System constants ##########
k = 0.1
g = 0.5
w = pi
pars = [k, g, w]

U0 = [0, 1]

######## Shooting method ##########
#U0 = shooting(Duffing, U0, 2*pi/w, pars)

######## Solve for one periodic orbit ########
#t = linspace(0, 2*pi / w , 500)
#x = scipy.integrate.odeint(Duffing, U0, t, args= (pars,))


########################################################################


#Results(Duffing, U0 , pars, 1, 0.01, 1120, 'odeint', 0) 


