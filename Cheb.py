## CHEB  compute D = differentiation matrix, x = Chebyshev grid

from numpy import *
from numpy.matlib import repmat
from numpy import transpose



def cheb(N):

    if N==0:
        D=0
        x=1
        return (D, x)
    x = transpose(cos(dot(pi, range(N))/N)) 
    c = array([(2), ones(N-1,1),  2])*(-1)**(range(N))
    X = repmat(x,1,N+1)
    dX = transpose(X-X)         
    D  = (c*transpose(1./c))/(dX+(eye(N+1)))      # off-diagonal entries
    D  = D - diag(sum(transpose(D)))                 # diagonal entries

    return (D, x)


        
        
def Cheb(N):
    if N == 0:
        D = 0
        x = 0
        return (D, x)
    x = array([])
    c = array(2)

    for i in range(N+1):
        if i <= N:
            x = append(x, cos( (pi * i ) / N  ))
        if i == 0:
            pass
        if i == N-1:
            
            c = append(c, 2*(-1)**(i-1))
        if i > N-2 :
            pass
            
        else:
            c = append(c, (-1)**(i-1))

    X = repmat(x,1, N+1)
    X = X.reshape(N +1, N +1)
    X = X.transpose() 
    dX = subtract(X , transpose(X))
    
    #print( divide(tensordot(c, (1/c).transpose() , axes = 0 ), dX + eye(N+1)) ) 
    
    D = divide(tensordot( c, (1/c).transpose(), axes = 0), (dX + eye(N+1)))


    
    D + D.reshape(N+1,N+1)
    print(diag(sum(D, axis = 1)))
    print(sum(D, axis = 1))
    D = D - diag(sum(D, axis = 1  ))

    #D = 0 
    return (D, x)

print(Cheb(5))

