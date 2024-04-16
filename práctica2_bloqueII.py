import matplotlib.pyplot as plt
import numpy as np
import numba
from matplotlib import animation
from scipy.integrate import solve_ivp,odeint

#Método de Verlet

Ms = 1.9891e30
Mt = 5.9722e24

GM = 6.6738e-11 * Ms

t_tot = 5 * 365 * 24 *3600
h = 3600
N = 5 * 365 * 24

def f(r):
    return -GM * r / np.dot(r,r)**1.5


r = np.zeros((N,2))
v = np.zeros((2*N,2))
Ep = np.zeros(N)
Ec = np.zeros(N)
E = np.zeros(N)


r[0,0] = 1.4719e11
v[0,1] =  3.0287e4
v[1] = v[0] + 1/2 * h * f(r[0])
Ep[0] = -GM * Mt / np.dot(r[0],r[0])**.5
Ec[0] = 1/2 * Mt * np.dot(v[0],v[0])
E[0] = Ep[0] + Ec[0]

for i in range(1,N):
    r[i] = r[i-1] + h*v[2*i - 1]
    k = h * f(r[i])
    v[2*i] = v[2*i - 1] + 1/2 * k 
    v[2*i + 1] = v[2*i - 1] + k
    Ep[i] = -GM * Mt / np.dot(r[i],r[i])**.5
    Ec[i] = 1/2 * Mt * np.dot(v[2*i],v[2*i])
    E[i] = Ep[i] + Ec[i]

#Radio en función del tiempo

plt.figure()
plt.plot(np.sqrt(r[:,0]**2 + r[:,1]**2))
plt.xlabel('t')
plt.ylabel('r')
plt.title('Radio en función del tiempo')
plt.show()

#Gráfica de x = f(y)
plt.figure()
plt.plot(r[:,0],r[:,1])
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc = 'best')
plt.show()

#Gráfica de las energías
plt.figure()
plt.plot(Ep,'r', label = 'Energía potencial')
plt.plot(Ec, 'b', label = 'Energía cinética')
plt.plot(E, 'g', label = 'Energía total')
plt.xlabel('t')
plt.ylabel('Energía')
plt.legend(loc = 'best')
plt.show()

#Variaciones de la energía total con respecto al tiempo
plt.figure()
plt.plot(E, 'g', label = 'Energía total')
plt.title('Energía total')
plt.xlabel('t')
plt.ylabel('Energía')
plt.show()

####################################################################################################
# =============================================================================
# Otro método
# =============================================================================

def fun(t, y):
    r = np.sqrt(y[0]**2 + y[1]**2)
    return [y[2], y[3], -GM * y[0]/r**3, -GM *y[1] /r**3]

def fun1(y,t):
    r = np.sqrt(y[0]**2 + y[1]**2)
    return [y[2], y[3], -GM * y[0]/r**3, -GM *y[1] /r**3]

# Condiciones iniciales [x, y, vx, vy] 
y0 = [1.4719e11,0, 0,3.0287e4]

t1 = np.arange(0,t_tot,3600)

solv = solve_ivp(fun,[0,t_tot],y0, method = 'RK45', t_eval = t1)
solOdeint = odeint(fun1,y0,t1)

#Radio en función del tiempo

plt.figure()
plt.plot(np.sqrt(solv.y[0,:]**2 + solv.y[1,:]**2))
plt.plot(np.sqrt(solOdeint[:,0]**2 + solOdeint[:,1]**2))
plt.xlabel('t')
plt.ylabel('r')
plt.title('Radio en función del tiempo')
plt.show()

plt.figure()
plt.plot(solv.y[0,:],solv.y[1,:])
plt.plot(solOdeint[:,0],solOdeint[:,1])
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc = 'best')
plt.show()

#############################################################################
#RUNGE-KUTTA-4

GM = 6.6738e-11*1.9891e30
m = 5.9722e24

def f(r):
    return -GM*r/np.dot(r,r)**1.5

t0 = 0.
t = 5*365*24*3600.
N = 5*365*24
h = 3600.

r = np.zeros((N,2))
v = np.zeros((N,2))
Ep = np.zeros(N)
Ec = np.zeros(N)
E = np.zeros(N)

r[0] = 1.4719e11 , 0
v[0] = 0 , 3.0287e4

Ec[0] = 1/2*m*(v[0,1])**2
Ep[0] = -GM*m/r[0,0]
E[0] = Ec[0] + Ep[0]

for i in range(0,N-1):
    k1 = h*v[i]
    l1 = h*f(r[i])
    k2 = h*(v[i]+l1/2)
    l2 = h*f(r[i]+k1/2)
    k3 = h*(v[i]+l2/2)
    l3 = h*f(r[i]+k2/2)
    k4 = h*(v[i]+l3)
    l4 = h*f(r[i]+k3)
    r[i+1] = r[i] + 1/6*(k1+2*k2+2*k3+k4)
    v[i+1] = v[i] + 1/6*(l1+2*l2+2*l3+l4)
    Ec[i+1] = 1/2*m*np.dot(v[i+1],v[i+1])
    Ep[i+1] = -GM*m/(np.dot(r[i+1],r[i+1]))**0.5
    E[i+1] = Ec[i+1] + Ep[i+1]

t1 = np.linspace(t0,t,N)
rad = np.zeros(N)

#Radio en función del tiempo
plt.figure()
plt.plot(t1,np.sqrt(r[:,0]**2 + r[:,1]**2))
plt.xlabel('radio')
plt.ylabel('tiempo')
plt.show()

#Trayectoria
plt.figure()
plt.plot(solv.y[0,:],solv.y[1,:])
plt.plot(solOdeint[:,0],solOdeint[:,1], label = 'odeint')
plt.plot(r[:,0],r[:,1],label= 'RK4')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc = 'best')
plt.show()

#Energias
plt.figure()
plt.plot(t1,Ec, label='Energía cinética')
plt.plot(t1,Ep, label='Energía potencial')
plt.plot(t1,E, label='Energía total')
plt.legend()
plt.show()

#Energia total
plt.figure()
plt.plot(E)
plt.show()















