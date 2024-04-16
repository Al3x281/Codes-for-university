import matplotlib.pyplot as plt
import numpy as np
import numba
from matplotlib import animation
from scipy.integrate import solve_ivp,odeint


n = 12
m = 6
sigm = 2.3151 #A
eps = 0.167 #ev
a = 3.603 #A

Nx=int(input('Número de celdas unidad en la dirección x: '))
Ny=int(input('Número de celdas unidad en la dirección y: '))
Nz=int(input('Número de celdas unidad en la dirección z: '))
Condicion = input('Condiciones de contorno (libre/periodicas): ')

def V(r):
    return 4*eps*((sigm/r)**n-(sigm/r)**m)

def redfcc(Nx,Ny,Nz):
    atom = 4*Nx*Ny*Nz
    base1 = np.array([0,0,0])
    base2 = np.array([a*0.5,a*0.5,0])
    base3 = np.array([a*0.5,0,a*0.5])
    base4 = np.array([0,a*0.5,a*0.5])
    posicion = []
    
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                posicion.append([a*i,a*j,a*k]+base1)
                posicion.append([a*i,a*j,a*k]+base2)
                posicion.append([a*i,a*j,a*k]+base3)
                posicion.append([a*i,a*j,a*k]+base4)

    red = np.array(posicion)          
    return red,atom 


red,atom = redfcc(Nx,Ny,Nz)

'''
#Mostramos en un gráfico 3D la red
fig=plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(red[:,0],red[:,1],red[:,2],'0')
ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.set_zlabel('Eje Z')
plt.show()
'''

#Sacamos las energías de cada partícula sin condiciones periódicas

def F(r):
    return -24*eps *sigm**6 * (r**6 - 2*sigm**6)/ r**13


def ener(Condicion, red,atom):
    V_v = np.zeros(atom)
    distmed = np.zeros(atom)
    f = np.zeros((atom,3))
    mod_f = np.zeros(atom)
    
    if Condicion == 'libre':
        for i in range(atom):
            for j in range(atom): 
                if i!=j:
                    distmed[j] = np.sqrt(np.dot(red[i]-red[j],red[i]-red[j]))
                    if distmed[j] <= 3*sigm:
                        V_v[i] += V(distmed[j])/2
                        f[i] +=  F(distmed[j]) * (red[i]-red[j])/distmed[j] * 0.5
                mod_f[i] = np.linalg.norm(f[i])
        Epp = np.sum(V_v)/atom
    
   

    if Condicion == 'periodicas':
        ejex = red[:,0]
        ejey = red[:,1]
        ejez = red[:,2]
        
        for i in range(atom):
            for j in range(atom):
                if i!=j:
                    deltx = ejex[i] - ejex[j]
                    delty = ejey[i] - ejey[j]
                    deltz = ejez[i] - ejez[j]
                    
                    if deltx > Nx*a/2:
                        deltx = deltx - Nx*a
                    
                    if deltx < -Nx*a/2:
                        deltx = deltx + Nx*a
                        
                    if delty > Ny*a/2:
                        delty = delty - Ny*a
                    
                    if delty < -Ny*a/2:
                        delty = delty + Ny*a
                        
                    if deltz > Nz*a/2:
                        deltz = deltz - Nz*a
                    
                    if deltz < -Nz*a/2:
                        deltz = deltz + Nz*a
                    
                    distmed[j] = np.sqrt(deltx**2 + delty**2 + deltz**2)
                    if distmed[j] <= 3*sigm:
                        V_v[i] += V(distmed[j])/2
                        f[i] +=  F(distmed[j]) * (red[i]-red[j])/distmed[j] * 0.5
                mod_f[i] = np.linalg.norm(f[i])
        '''
        for j in range(atom-1):
            deltx = ejex[-1] - ejex[j]
            delty = ejey[-1] - ejey[j]
            deltz = ejez[-1] - ejez[j]
            
            if deltx > Nx*a/2:
                deltx = deltx - Nx*a
            
            if deltx < -Nx*a/2:
                deltx = deltx + Nx*a
                
            if delty > Ny*a/2:
                delty = delty - Ny*a
            
            if delty < -Ny*a/2:
                delty = delty + Ny*a
                
            if deltz > Nz*a/2:
                deltz = deltz - Nz*a
            
            if deltz < -Nz*a/2:
                deltz = deltz + Nz*a
            
            distmed[j] = np.sqrt(deltx**2 + delty**2 + deltz**2)
            if distmed[j] <= 3*sigm:
                V_v[-1] += V(distmed[j])/2
        '''
        Epp = np.sum(V_v)/atom #energía por partícula
        
    return V_v, Epp, mod_f, f
                
                

V_v, Epp,mod_f,f= ener(Condicion,red,atom)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
red1 = ax.scatter(red[:,0],red[:,1],red[:,2], c= V_v, s = 30)
fig.colorbar(red1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.quiver(red[:,0],red[:,1],red[:,2], f[:,0],f[:,1],f[:,2], length=1, normalize=True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

# =============================================================================
# EXTRA 1
# =============================================================================


Temp = 300
vs = (np.random.rand(atom,3) - 0.5) * 10e-10
m = 63.55 * 931493614.8389 / (3e8)**2
kb = 8.6181024 * 1e-5
mod_v = np.zeros(atom)

for i in range(atom):
    mod_v[i] = np.linalg.norm(vs[i])

Ec = np.sum(1/2 * m * mod_v**2)

Trand = 2/3 * Ec/(kb * atom)

#Factor de escala
sc = np.sqrt(Temp/Trand)

v_esc = sc*vs
mod_v1 = np.zeros(atom)
for i in range(atom):
    mod_v1[i] = np.linalg.norm(v_esc[i])

Ec1 = np.sum(1/2 * m * mod_v1**2)

T = 2/3 * Ec1/(kb * atom)


# =============================================================================
# EXTRA 2
# =============================================================================
'''
Npasos = 100

h = 10e-5

def y(r):
    return ener(Condicion,r,atom)[3]/m

@numba.jit
def verlet(Npasos,h,red):
    r = np.zeros((Npasos,atom,3))
    r[0] = red
    v = np.zeros((2*Npasos,atom,3))
    v[0] = v_esc

    
    for i in range(1,Npasos):
        r[i] = r[i-1] + h*v[2*i - 1]
        k = h * y(r[i])
        v[2*i] = v[2*i - 1] + 1/2 * k 
        v[2*i + 1] = v[2*i - 1] + k
        
    return r,v
    
r, v = verlet(Npasos,h,red)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(r[:,4,0],r[:,4,1],r[:,4,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

'''








