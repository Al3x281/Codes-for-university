import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import numba
from matplotlib import animation
import pandas as pd

Nx, Ny, Nz = 5,5,5


def red_cristalina(Nx,Ny,Nz, tipo):
    
    if tipo == 'cubica':
        Natom = Nx*Ny*Nz
        pos1 = []
        base = np.array([0,0,0])
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    pos1.append([i+base[0],j+base[1],k+base[2]])
        
        pos1 = np.array(pos1)
        return pos1
    
    if tipo == 'cubica2':
        x = np.arange(0,Nx,1)
        y = np.arange(0,Ny,1)
        z = np.arange(0,Nz,1)
        xx,yy,zz = np.meshgrid(x,y,z)
        pos = np.array([xx,yy,zz])
        return pos
        
    
    if tipo == 'BCC':
        Natom = 2* Nx*Ny*Nz
        pos2 = []
        base = np.array([[0,0,0],[0.5,0.5,0.5]])
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    pos2.append([i+base[0,0],j+base[0,1],k+base[0,2]])
                    pos2.append([i+base[1,0],j+base[1,1],k+base[1,2]])
        pos2 = np.array(pos2)
        return pos2
    
    if tipo == 'BCC2':
        base = np.array([[0,0,0],[0.5,0.5,0.5]])
        x1,y1,z1 = np.zeros((len(base),Ny,Nx,Nz)), np.zeros((len(base),Ny,Nx,Nz)), np.zeros((len(base),Ny,Nx,Nz))
        for i in range(len(base)):
            x1[i],y1[i],z1[i] = np.meshgrid(np.arange(base[i,0],base[i,0]+Nx,1),np.arange(base[i,1],base[i,1]+Ny,1),np.arange(base[i,2],base[i,2]+Nz,1))
        x,y,z = np.concatenate((x1[:]), axis=0), np.concatenate((y1[:]), axis=0), np.concatenate((z1[:]), axis=0)
        pos = np.array([x,y,z])
        return pos
        
    if tipo == 'FCC':
        Natom = 4* Nx*Ny*Nz
        pos3 = []
        base = np.array([[0,0,0],[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]])
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    pos3.append([i+base[0,0],j+base[0,1],k+base[0,2]])
                    pos3.append([i+base[1,0],j+base[1,1],k+base[1,2]])
                    pos3.append([i+base[2,0],j+base[2,1],k+base[2,2]])
                    pos3.append([i+base[3,0],j+base[3,1],k+base[3,2]])
        pos3 = np.array(pos3)
        return pos3
    
    if tipo == 'FCC2':
        base = np.array([[0,0,0],[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]])

        x1,y1,z1 = np.zeros((len(base),Ny,Nx,Nz)), np.zeros((len(base),Ny,Nx,Nz)), np.zeros((len(base),Ny,Nx,Nz))
        for i in range(len(base)):
            x1[i],y1[i],z1[i] = np.meshgrid(np.arange(base[i,0],base[i,0]+Nx,1),np.arange(base[i,1],base[i,1]+Ny,1),np.arange(base[i,2],base[i,2]+Nz,1))
        x,y,z = np.concatenate((x1[:]), axis=0), np.concatenate((y1[:]), axis=0), np.concatenate((z1[:]), axis=0)
        pos = np.array([x,y,z])
        return pos
        
    if tipo == 'diamante':
        Natom = 8* Nx*Ny*Nz
        pos4 = []
        base = np.array([[0,0,0],[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5], \
                         [0.25,0.25,0.25],[0.75,0.75,0.25],[0.75, 0.25, 0.75],[0.25, 0.75, 0.75]])
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    pos4.append([i+base[0,0],j+base[0,1],k+base[0,2]])
                    pos4.append([i+base[1,0],j+base[1,1],k+base[1,2]])
                    pos4.append([i+base[2,0],j+base[2,1],k+base[2,2]])
                    pos4.append([i+base[3,0],j+base[3,1],k+base[3,2]])
                    pos4.append([i+base[4,0],j+base[4,1],k+base[4,2]])
                    pos4.append([i+base[5,0],j+base[5,1],k+base[5,2]])
                    pos4.append([i+base[6,0],j+base[6,1],k+base[6,2]])
                    pos4.append([i+base[7,0],j+base[7,1],k+base[7,2]])
        pos4 = np.array(pos4)
        return pos4
    
    if tipo == 'diamante2':
        base = np.array([[0,0,0],[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5], \
                         [0.25,0.25,0.25],[0.75,0.75,0.25],[0.75, 0.25, 0.75],[0.25, 0.75, 0.75]])
        x1,y1,z1 = np.zeros((len(base),Ny,Nx,Nz)), np.zeros((len(base),Ny,Nx,Nz)), np.zeros((len(base),Ny,Nx,Nz))
        for i in range(len(base)):
            x1[i],y1[i],z1[i] = np.meshgrid(np.arange(base[i,0],base[i,0]+Nx,1),np.arange(base[i,1],base[i,1]+Ny,1),np.arange(base[i,2],base[i,2]+Nz,1))
        x,y,z = np.concatenate((x1[:]), axis=0), np.concatenate((y1[:]), axis=0), np.concatenate((z1[:]), axis=0)
        pos = np.array([x,y,z])
        return pos
    return

'''
El codigo se ha realizado de dos formas diferentes, la primera aquella con bucles que para grandes valores de 
Nx,Ny,Nz es más lenta. Y la otra utilizando meshgrids que es mucho más rápida, pero no sé como hacer el 
archivo para que me guarde las posiciones ya que estoy usando arrays de 4 dimensiones. Por ello, para hacer el
archivo uso el otro método.
'''
    
    
Nx, Ny, Nz = 1,1,1     
pos = red_cristalina(Nx,Ny,Nz, 'diamante2')

'''
#Para graficar todas las que no llevan indice 1
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(pos[:,0],pos[:,1],pos[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()
'''
#Para graficar las que llevan indice 2
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(pos[0],pos[1],pos[2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()
          

'''
#Para guardar el archivo con las posiciones
column_names = ['X', 'Y', 'Z']
df = pd.DataFrame(pos, columns=column_names)
df.to_csv('./ejemplo.txt', sep=' ',  index_label='Atomo')

with open('ejemplo.txt', 'r') as file:
    # Lee el contenido actual del archivo
    contenido_existente = file.read()

# Abre el archivo en modo de escritura (w) para sobrescribirlo
with open('ejemplo.txt', 'w') as file:
    # Escribe la nueva línea de texto al principio del archivo
    nueva_linea = "El numero de atomos es {}".format(len(pos))
    file.write(nueva_linea + '\n')
    
    # Luego, escribe el contenido original que habías leído previamente
    file.write(contenido_existente)
'''
'''
Nx, Ny, Nz = 4,3,1 
Natom = 2* Nx*Ny*Nz
pos3 = []
base = np.array([[1,0,0],[0.5,np.sqrt(3)/2,0]])


for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            pos3.append([i+base[0,0],2*j+base[0,1],base[0,2]])
            pos3.append([i+base[1,0],2*j+base[1,1],base[1,2]])
            #pos3.append([i+base[2,0],j+base[2,1],base[2,2]])
            #pos3.append([i+base[3,0],j+base[3,1],k+base[3,2]])
            #pos3.append([i+base[4,0],j+base[4,1],k+base[4,2]])
            #pos3.append([i+base[5,0],j+base[5,1],k+base[5,2]])

pos3 = np.array(pos3)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(pos3[:,0],pos3[:,1],pos3[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(0,5)
ax.set_ylim(0,5)
plt.show()

base = np.array([[1,0,0],[0.5,np.sqrt(3)/2,0]])
x1,y1,z1 = np.zeros((len(base),Nx,Ny,Nz)), np.zeros((len(base),Nx,Ny,Nz)), np.zeros((len(base),Nx,Ny,Nz))
for i in range(len(base)):
    x1[i],y1[i],z1[i] = np.meshgrid(np.arange(base[i,0],base[i,0]+Nx,1),np.arange(base[i,1],base[i,1]+2*Ny,2),np.arange(base[i,2],base[i,2]+Nz,1))
x,y,z = np.concatenate((x1[:]), axis=0), np.concatenate((y1[:]), axis=0), np.concatenate((z1[:]), axis=0)
pos = np.array([x,y,z])

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(pos[0],pos[1],pos[2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(0,5)
ax.set_ylim(0,5)
plt.show()
'''

'''
base = np.array([[0,0,0],[0.5,0.5,0.5]])

x,y,z = np.meshgrid(np.arange(0,Nx,1),np.arange(0,Ny,1),np.arange(0,Nz,1))

x1,y1,z1 = [],[],[]
for i in range(len(base)):
    x1.append(x.flatten('C') + base[i,0])
    y1.append(y.flatten('C') + base[i,1])
    z1.append(z.flatten('C') + base[i,2])

pos = np.array(x1), np.array(y1), np.array(z1)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(pos[0],pos[1],pos[2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()
'''