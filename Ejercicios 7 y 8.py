import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import block_diag
from matplotlib import animation
from scipy.linalg import solve_banded, solveh_banded
from scipy.sparse import diags, identity

#Advección

accel = 10
def animate_ondas(x,sol,N,step):
    fig, ax = plt.subplots()
    
    def update(frame):
        ax.cla()
        ax.plot(x, sol[frame*accel])
        ax.set_title('Solucion')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x)')
        #ax.set_ylim(np.min(sol),np.max(sol))
    
    ani = animation.FuncAnimation(fig, update, frames=range(int(step/(accel))), interval = 10, repeat=False)
    
    plt.show()
    return ani


# Parámetros
L = 10.0  # Longitud del dominio espacial
T = 10.  # Tiempo total de simulación
c = 1.   # Velocidad de advección
Nx = 1000  # Número de puntos de malla en el espacio
Nt = 1000 # Número de pasos de tiempo
dx = L / (Nx)
dt = T / Nt

# Inicialización de la función u(x, t=0)
x = np.linspace(0, L, Nx)
u0 = np.exp(-10 * (x - 1)**2)

def advec(c,T,Nt,u0,metodo):
    u = np.zeros((Nt,Nx))
    dt = T / Nt
    dx = L / (Nx - 1)
    u[0,:] = u0
    Cr = c*dt/dx
    if metodo == 'upwind':
        for i in range(1,Nt):
            u[i,1:] = ((1-Cr)*u[i-1,1:] + Cr*u[i-1,0:-1])
            u[i,0] = u[i,-1]
        return u
    if metodo == 'downwind':
        for i in range(1,Nt):
            u[i,:-1] = (1-Cr)*u[i-1,:-1] + Cr*u[i-1,1:]
            u[i,-1] = u[i,0]
        return u
    if metodo == 'central':
        for i in range(1,Nt):
            u[i,1:-1] = (1-Cr/2)*u[i-1,2:] + Cr/2 * u[i-1,0:-2]
            u[i,0] = u[i,-2]
            u[i,-1] = u[i,1]
        return u
    return

u = advec(c, T, Nt, u0, 'upwind')
#ani4 = animate_ondas(x, u, Nx, T/dt)


###################################################################################################

#Burguers

L = 10.0  # Longitud del dominio espacial
T = 1.0   # Tiempo total de simulación
Nx = 100  # Número de puntos de malla en el espacio
Nt = 1000  # Número de pasos de tiempo
dx = L / (Nx - 1)
dt = T / Nt

# Inicialización de la función u(x, t=0)
x = np.linspace(0, L, Nx)
u0 = 3 * np.sin(2 * np.pi * x / L)


#Upwind
def burguers(T,Nt,u0):
    u = np.zeros((Nt,Nx))
    dt = T / Nt
    dx = L / (Nx - 1)
    u[0,:] = u0
    Cr = dt/dx
    for i in range(1,Nt):
        for j in range(1,Nx):#Upwind
            if u[i][j]>=0:
                u[i][j] = -Cr*u[i-1][j]*(u[i-1][j]-u[i-1][j-1])+u[i-1][j]
                u[i][0]=u[i][-2] #Condiciones periódicas
                u[i][-1]=u[i][1]
            if u[i][j]<0:#Upwind invertido
                u[i][j] = -Cr*u[i-1][j]*(u[i-1][j+1]-u[i-1][j])+u[i-1][j]
                u[i][0]=u[i][-2]#condiciones periódicas
                u[i][-1]=u[i][1]
    return u
    


u = burguers(T, Nt, u0)
#ani5 = animate_ondas(x, u, Nx, T/dt)

plt.plot(u[50,:])








