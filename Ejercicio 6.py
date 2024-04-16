import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import block_diag
from matplotlib import animation
from scipy.linalg import solve_banded, solveh_banded
from scipy.sparse import diags, identity

#Ejercicio 6


#Creamos la función que grafique las soluciones. Es una modificación del ejercicio anterior 
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


"""
Método explícito
"""
def ondas_explicito(kappa,rho, x, u1,u0):
    dx = 1/N    
    c = 0.5
    T = 0
    i = 0
    
    us = []
    us.append(u0)
    us.append(u1)
    u = np.zeros(N)
    
    while T < 50:
        u[1:N-1] = (c**2*dt**2/dx**2)*(us[i+1][2:] -2*us[i+1][1:N-1] + us[i+1][0:N-2]) \
            - (2*kappa*dt)/rho * (us[i+1][1:N-1] - us[i][1:N-1]) + 2*us[i+1][1:N-1] \
                - us[i][1:N-1]
        us.append(u.copy())
        T += dt
        i += 1

    us = np.array(us)
    return us

N = 100

x = np.linspace(0,1,N)
u0 = np.ones(N)*np.sin(3*np.pi*x)
u1 = u0


dt = 1e-2
rho = 1e-2
kappa = 1e-3



us = ondas_explicito(kappa, rho, x, u1, u0)
#ani = animate_ondas(x, us, N, 50/dt)


"""
Método implícito
"""

N = 100

x = np.linspace(0,1,N)
u0 = np.ones(N)*np.sin(3*np.pi*x)
u1 = u0

dx = 1/N
dt = 1e-3
rho = 1e-2
kappa = 1e-3
c = 0.5


def ondas_implicito(kappa,rho, x, u1,u0):
    d1 = np.ones(N)*(1 + 2*kappa/rho*dt + 2*c**2*dt**2/dx**2)
    o1 = np.ones(N-1)*(-c**2*dt**2/dx*2)
    
    M = sparse.diags([d1, o1, o1], [0,-1,1],shape=(N,N)).toarray()
    M2 = np.linalg.inv(M)
    T = 0
    i = 1
    
    us = [u0,u1]
    
    while T < 40:
        us.append(np.dot(M2,(2*us[i] - us[i-1] +2*kappa*dt/rho*us[i])))
        i += 1
        T += dt
    
    us = np.array(us)
    return us

us1 = ondas_implicito(kappa, rho, x, u1, u0)
#ani2 = animate_ondas(x, us, N, 40/dt)



####################################################################################################

#Telégrafo

def telegrafo(x, u1,u0):
    dx = 1/N    
    c = 0.5
    T = 0
    i = 0
    
    us = []
    us.append(u0)
    us.append(u1)
    u = np.zeros(N)
    
    while T < 50:
        u[1:N-1] = (c**2*dt**2/dx**2)*(us[i+1][2:] -2*us[i+1][1:N-1] + us[i+1][0:N-2]) \
            - dt * (us[i+1][1:N-1] - us[i][1:N-1]) + 2*us[i+1][1:N-1] \
                - us[i][1:N-1] - 2*dt**2 * us[i+1][1:N-1]
        us.append(u.copy())
        T += dt
        i += 1

    us = np.array(us)
    return us

N = 100

x = np.linspace(0,1,N)
u0 = np.ones(N)*np.sin(3*np.pi*x)
u1 = u0

dx = 1/N
dt = 1e-3
c = 0.5
us3 = telegrafo(x, u1, u0)
#ani3 = animate_ondas(x, us3, N, 50/dt)

##################################################################################################
    






