import matplotlib.pyplot as plt
import numpy as np
import numba
from matplotlib import animation
from scipy.integrate import solve_ivp,odeint


def V(r,eps,sig,n,m):
    return 4*eps* ((sig/r)**n - (sig/r)**m)


def fcc(Nx,Ny,Nz,a):
    base = a* np.array([[0,0,0],[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]])

    x1,y1,z1 = np.zeros((len(base),Ny,Nx,Nz)), np.zeros((len(base),Ny,Nx,Nz)), np.zeros((len(base),Ny,Nx,Nz))
    for i in range(len(base)):
        x1[i],y1[i],z1[i] = np.meshgrid(np.arange(base[i,0],base[i,0]+a*Nx,a),np.arange(base[i,1],base[i,1]+a*Ny,a),np.arange(base[i,2],base[i,2]+a*Nz,a))
    x,y,z = np.concatenate((x1[:]), axis=0), np.concatenate((y1[:]), axis=0), np.concatenate((z1[:]), axis=0)
    x2,y2,z2 = x.flatten('C'), y.flatten('C'),z.flatten('C')
    pos = np.array([x2,y2,z2]).T
    return pos

def fcc2(Nx,Ny,Nz,a):
    Natom = 4* Nx*Ny*Nz
    pos3 = []
    base = a*np.array([[0,0,0],[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]])
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                pos3.append([a*i+base[0,0],a*j+base[0,1],a*k+base[0,2]])
                pos3.append([a*i+base[1,0],a*j+base[1,1],a*k+base[1,2]])
                pos3.append([a*i+base[2,0],a*j+base[2,1],a*k+base[2,2]])
                pos3.append([a*i+base[3,0],a*j+base[3,1],a*k+base[3,2]])
    pos3 = np.array(pos3)
    return pos3


Nx,Ny,Nz = 5,5,5
n = 12
m = 6
eps = 0.167
sig = 2.3151
a = 3.603
pos = fcc2(Nx,Ny,Nz, a) 

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(pos[0],pos[1],pos[2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()
'''
# =============================================================================
# Energia sin condiciones periódicas
# =============================================================================

N = 4*Nx*Ny*Nz 
dist = np.zeros(N)
E = np.zeros(N)


for i in range(N-1):
    for j in range(N):
        if i!=j:
            dist[j] = np.sqrt(np.dot(pos[i]-pos[j],pos[i]-pos[j]))
            if dist[j] <= 3*sig:
                E[i] += V(dist[j],eps,sig,n,m)/2
                
    
    
Etot = np.sum(E)/N 
    

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(pos[:,0],pos[:,1],pos[:,2], c= E, s = 100, cmap = 'viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()









