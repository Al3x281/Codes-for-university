import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import block_diag
from matplotlib import animation
from scipy.linalg import solve_banded, solveh_banded

#Al ejecutar, ir quitando los comentarios de las animaciones, porque si se ejecutan todas a la vez, a mi por lo 
#menos, se me cuelga el Spyder

def difusion(x,T,N,D,step):
    dx = (x[-1]-x[0])/N #definimos el paso en x
    dt = dx**2 / (4 * D) #ajustamos el tiempo con la condición de convergencia
    T0 = np.zeros((step,N)) #inicializamos el array donde guardaremos todos los pasos
    T0[0,:] = T #Establecemos las condiciones iniciales con una temperatura fija
    cont = 0
    while True:
        T[1:N-1] += D*dt/(dx**2) * (T[2:N] + T[0:N-2] - 2*T[1:N-1])
        cont += 1
        T0[cont-1,:] = T
        if cont == step:
            break
    
    return T0

#Creamos la función para graficar
def animate_difusion(x,sol,T, N, D, step):
    #S = np.zeros((1,N))
    #S[0,:] = sol[0]
    fig, ax = plt.subplots()
    
    def update(frame):
        #S[0,:] = sol[frame*accel]
        ax.cla()
        ax.plot(x, sol[frame*accel])
        #ax.imshow(S, cmap = 'hot', aspect = 'auto') #Para ver como una barra de metal en la que desciende la temperatura
        ax.set_title('Solucion')
        ax.set_xlabel('X')
        ax.set_ylabel('Temperatura')
        ax.set_ylim(np.min(sol),np.max(sol)+3)
    
    ani = animation.FuncAnimation(fig, update, frames=range(int(step/(accel))), interval = 10, repeat=False)
    
    plt.show()
    return ani

#Establecemos los valores de las constantes y condiciones del problema
N = 100
step = 10000
D = 10e-2
T = 100 * np.ones(N)
T[0], T[-1] = 0,0
x = np.linspace(0, 1, N)
accel = 10
sol = difusion(x,T, N, D, step)
#anim = animate_difusion(x, sol,T, N, D, step)


###############################################################################################

#Difusion con C-N

def difusion_CN(x,T,D,N,step):
    dx = (x[-1]-x[0])/N
    dt = dx**2 / (4 * D)
    
    #Definimos la primera matriz
    upper1 = -((D*dt)/(2 * dx**2))*np.ones(N)
    upper1[0],upper1[1] = 0,0
    lower1 = -((D*dt)/(2 * dx**2))*np.ones(N)
    lower1[-1],lower1[-2] = 0,0
    diag1 = (1+ ((D*dt)/(dx**2)))*np.ones(N)
    diag1[0],diag1[-1] = 1,1
    #Creamos la matriz como los tres array independientes porque se usará el algoritmo para 
    #resolver matrices tridiagonales
    A = np.array([upper1,diag1,lower1])
    
    #Definimos la segunda
    upper2 = ((D*dt)/(2 * dx**2))*np.ones(N)
    lower2 = ((D*dt)/(2 * dx**2))*np.ones(N)
    diag2 = (1 - ((D*dt)/(dx**2)))*np.ones(N)
    B = sparse.diags([diag2, lower2, upper2], [0,-1,1],shape=(N,N)).toarray()

    T0 = np.zeros((step,N))
    T0[0,:] = T
    cont = 0
    #Resolvemos el sistema
    while True:
        b = np.dot(B,T)
        b[0],b[-1] = 0,50
        R = solve_banded((1, 1), A, b) #Es el algoritmo de Thomas de la libreria scipy
        cont += 1
        T0[cont-1,:] = R
        R,T = T,R
        if cont == step:
            break
    
    return T0

T = 100 * np.ones(N)
T[0], T[-1] = 0, 50
x = np.linspace(0,1,N)
R = difusion_CN(x,T, D, N,step)
#anim1 = animate_difusion(x,R,T, N, D, step)


##################################################################################################
#Con diferente coeficiente de difusión

def difusion_CN_diff(x,T,D1,D2,N,step):
    dx = (x[-1]-x[0])/N
    dt = 1e-2
    
    #Definimos la primera matriz
    upper1 = -((D1*dt)/(2 * dx**2))*np.ones(N)
    lower1 = -((D1*dt)/(2 * dx**2))*np.ones(N)
    diag1 = (1+ ((D1*dt)/(dx**2)))*np.ones(N)
    diag1[0],diag1[-1] = 1,1
    A = sparse.diags([diag1, lower1, upper1], [0,-1,1],shape=(N,N)).toarray()
    A[int(0.4*N):int(0.6*N),int(0.4*N):int(0.6*N)] = sparse.diags([(1+ ((D2*dt)/(dx**2)))*np.ones(N), -((D2*dt)/(2 * dx**2))*np.ones(N), -((D2*dt)/(2 * dx**2))*np.ones(N)], [0,-1,1],shape=(int(0.6*N)-int(0.4*N),int(0.6*N)-int(0.4*N))).toarray()
    A[int(0.4*N),int(0.4*N)-1] =  A[int(0.6*N)-1,int(0.6*N)] = -((D2*dt)/(2 * dx**2))
    A[-1,-2] = A[0,1] = 0
    
    #Definimos la segunda
    upper2 = ((D1*dt)/(2 * dx**2))*np.ones(N)
    lower2 = ((D1*dt)/(2 * dx**2))*np.ones(N)
    diag2 = (1 - ((D1*dt)/(dx**2)))*np.ones(N)
    B = sparse.diags([diag2, lower2, upper2], [0,-1,1],shape=(N,N)).toarray()
    B[int(0.4*N):int(0.6*N),int(0.4*N):int(0.6*N)] = sparse.diags([(1 - ((D2*dt)/(dx**2)))*np.ones(N), ((D2*dt)/(2 * dx**2))*np.ones(N), ((D2*dt)/(2 * dx**2))*np.ones(N)], [0,-1,1],shape=(int(0.6*N)-int(0.4*N),int(0.6*N)-int(0.4*N))).toarray()
    B[int(0.4*N),int(0.4*N)-1] = B[int(0.6*N)-1,int(0.6*N)] = ((D2*dt)/(2 * dx**2))
    
    T0 = np.zeros((step,N))
    T0[0,:] = T
    cont = 0
    C = np.linalg.inv(A)
    #Resolvemos el sistema
    while True:
        b = np.dot(B,T)
        b[0],b[-1] = 0,50
        R = np.dot(C,b) 
        cont += 1
        T0[cont-1,:] = R
        R,T = T,R
        if cont == step:
            break
    
    return T0

accel = 1
N = 100
step = 100000
D1,D2 = 1e-2, 1e-3
T = 100 * np.ones(N)
T[0], T[-1] = 0, 50
x = np.linspace(0,1,N)
M = difusion_CN_diff(x,T, D1,D2, N,step)
anim2 = animate_difusion(x,M,T, N, D, step)



######################################################################################################
#Schrodinger con C-N


def Scro_CN(phi,N,V):
    x = np.linspace(0,1,N)
    dx = (x[-1]-x[0])/N
    dt = dx**2 / (4)
    
    #Definimos la primera matriz
    upper1 =  dt/(4*dx**2) *np.ones(N)
    lower1 =  dt/(4*dx**2) *np.ones(N)
    diag1 = (-dt/(2*dx**2) + 1j - dt/(2) * V) 
    #A = np.array([upper1,diag1,lower1])
    A = sparse.diags([diag1, lower1, upper1], [0,-1,1],shape=(N,N)).toarray() #Esta matriz se puede usar si se utiliza otro método de resolución
    
    #Definimos la segunda matriz
    
    upper2 = - dt/(4*dx**2) *np.ones(N)
    lower2 =  - dt/(4*dx**2) *np.ones(N)
    diag2 = (dt/(2*dx**2) + 1j + dt/(2) * V)
    B = sparse.diags([diag2, lower2, upper2], [0,-1,1],shape=(N,N)).toarray()
    
    phi0 = np.zeros((int((step))+1,N), dtype='complex128')
    phi0[0,:] = phi
    cont = 0
    while True:
        b = np.dot(B,phi)
        R = solve_banded((1,1),A,b)
        cont += 1
        R,phi = phi,R
        phi0[int(cont),:] = R
        if cont == step:
            break
    
    return phi0


def animate_schro(x,sol,T, N, D, step):
    S = np.zeros((1,N))
    S[0,:] = sol[0]
    fig, ax = plt.subplots()
    
    def update(frame):
        S[0,:] = sol[frame*accel]
        ax.cla()
        ax.plot(x, abs(sol[frame*accel]))
        #ax.imshow(S, cmap = 'Greens', aspect = 'auto')
        ax.set_title('Solucion')
        ax.set_xlabel('X')
        ax.set_ylabel('($|\\psi|^2$)')
        #ax.set_ylim(np.min(sol),np.max(sol))
    
    ani = animation.FuncAnimation(fig, update, frames=range(int(step/(accel)-accel*2)), interval = 10, repeat=False)
    
    plt.show()
    return ani


#Pozo infinito
step = 10000 
N = 100
x = np.linspace(0,1,N, dtype = 'complex128' )
V = np.zeros(N, dtype='complex128')
V[-1], V[0] = 1e5,1e5
phi =  np.sin(3* np.pi * x) * np.exp(1j*np.pi*x)
phi_s = Scro_CN(phi, N, V)


#anim4 = animate_schro(x,phi_s,T, N, D, step)

#Oscilador armónico

step = 10000  
N = 100
k = 1e-5
x = np.linspace(-1,1,N)
V = np.ones(N, dtype='complex128') * 1/2 * k * x**2
phi = (8*x**3 -12*x) * np.exp(-x**2)
phi_s = Scro_CN(phi, N, V)



#anim5 = animate_schro(x,phi_s,T, N, D, step)









