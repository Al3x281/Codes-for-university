import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import numba
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure

#Definimos en primer lugar los parámetros que vamos a usar durante toda la práctica.

N = 25     #Tamaño de la celda de simulación.
J = 1
T = 0.5    #Temperatura normalizada en unidades de la cte de Boltzmann T = J/K_b.
it = 8000  #Número de iteraciones.
accel = 5  #Valor que nos permite acelerar la animación.
H = 0     #Valor del campo magnético en unidades de mu. H = T/mu

#Generamos la matriz con valores 1 y -1 de forma aleatoria.

@numba.jit
def Ising(N):
    M = np.zeros([N+2,N+2])
    for i in range(N+2):
        for j in range(N+2):
            M[i,j] = np.random.choice([-1,1], p = [0.5,0.5])

    M[0,:] = M[N,:]
    M[N+1,:] = M[1,:]
    M[:,0] = M[:,N]
    M[:,N+1] = M[:,1]
    return M

#Otra forma de hacer Ising sin bucles.
def Ising_1(N):
    init_random = np.random.random((N+2,N+2))
    M = np.zeros((N+2, N+2))
    M[init_random>=0.5] = 1
    M[init_random<0.5] = -1
    M[0,:] = M[N,:]
    M[N+1,:] = M[1,:]
    M[:,0] = M[:,N]
    M[:,N+1] = M[:,1]
    return M

#Calculamos la energía de la matriz teniendo en cuenta la interacción de los primeros 
#vecinos.

@numba.jit
def Energia(M,J,H):
    E = 0
    Mag = np.sum(M)
    for i in range(1,N):
        for j in range(1,N):
            E += M[i,j]*M[i-1,j] + M[i,j]*M[i+1,j] + M[i,j]*M[i,j-1] + M[i,j]*M[i,j+1]
    E1 = -J/2 * E - H*Mag
    
    return E1,Mag

#He encontrado esta forma de calcular las energías. Dejo el anterior que es el hecho por mí, 
#pero utilizo este que es mucho más rapido. Utiliza una convolución basándose en la matriz kern
#que establece cuales son los primeros vecinos.

def Energia2(M): 
    kern = generate_binary_structure(2, 1) 
    kern[1][1] = False
    arr =  M * convolve(M, kern, mode='constant', cval=0)
    Mag = np.sum(M)
    return arr.sum(),Mag

#Generamos una variación de un spin cualquiera de la matriz de spines generada anteriormente

def Ising2(M):
    H,K = np.random.randint(1,N+1),np.random.randint(1,N+1)
    M2 = np.copy(M)

    if M2[H,K] == -1:
        M2[H,K] = 1
    else:
        M2[H,K] = -1
   
    M2[0,:] = M2[N,:]
    M2[N+1,:] = M2[1,:]
    M2[:,0] = M2[:,N]
    M2[:,N+1] = M2[:,1]
    return M2

#Con la función siguiente, decidimos si nos quedamos con la matriz anterior o la siguiente
#dependiendo del balance energético. (Cabe destacar que en este programa no se ha tenido en 
#cuenta la eficiencia del programa y simplemente se calculan las energías de nuevo a cada 
#paso.)

@numba.jit
def MonteCarlo(it,T):
    beta = 1/T
    X = Ising(N)
    Z = np.copy(X)
    E = []
    matrices = []
    mag = []
    for i in range(it):
        Y = Ising2(Z)
        E1 = Energia2(Z)[0]
        E2 = Energia2(Y)[0]
        M1 = Energia2(Z)[1]
        M2 = Energia2(Y)[1]
        delta = E2 - E1
        pnew = 1/(1+ np.exp(beta*delta))
        pold = 1/(1+ np.exp(-beta*delta))
        w = np.random.choice([0,1], p=[pnew,pold])
        if w == 0:
            Z = np.copy(Y)
            E.append(E2)
            mag.append(M2)
        else:
            E.append(E1)
            mag.append(M1)
        matrices.append(X)
    matrices1 = np.array(matrices)
    return Z,E,matrices1,mag

X,E,M,Mag = MonteCarlo(it,T)[0], MonteCarlo(it,T)[1], MonteCarlo(it,T)[2], MonteCarlo(it,T)[3]


############################################################################################

#Ahora vamos a crear la animación

cmap1 = plt.cm.flag #winter, afmhot,gray,gist_heat,copper,Wistia,cool, inferno
fig = plt.figure()
image = plt.imshow(M[0,:,:],cmap = cmap1)
plt.title('Módelo de Ising en 2D')
plt.xlabel('X')
plt.ylabel('Y')

def updatefig(p):
    global matrices
    image.set_array(M[p*accel,:,:])
    return image,

ani = animation.FuncAnimation(fig,updatefig, frames=int(it/accel), interval = 20, repeat = True)
plt.show()

###############################################################################################

#Vamos a crear un Slider para poder variar a nuestro gusto el número de iteraciones y no
#tener que depender del número de frames de la animación. Así, podemos el frame que queramos.
init_frequency = 1

#fig = plt.figure()
fig, ax = plt.subplots()
line = plt.imshow(M[init_frequency-1,:,:],cmap = cmap1)
#line, = plt.imshow(M[-1,:,:],cmap = cmap1)
ax.set_xlabel('Mapa de spines')

# Ajustamos la gráfica para poder colocar el slider
fig.subplots_adjust(left=0.25, bottom=0.25)

# Creamos el Slider horizontal para cambiar los valores.
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='Iteraciones',
    valmin=1,
    valmax=it,
    valstep= 1,
    valinit=init_frequency,
    color="green"
)

# La función para cambiar los valores en función de los valores que tome el slider.
def update(val):
    line.set_data(M[int(val)-1,:,:])
    fig.canvas.draw_idle()
    return line,


# Se registra el cambio de la función con cada valor del slider
freq_slider.on_changed(update)

plt.show()

###################################################################################


#Vamos a graficar ahora la energía y la magnetizacion en funcion del numero de iteraciones.
x = np.linspace(0,it,it)


plt.figure()
plt.plot(x,E,label = 'Energía')
plt.plot(x,Mag,'r',label = 'Magnetización')
plt.xlabel('Iteraciones')
plt.legend(loc = 'best')
plt.show()

###################################################################################
'''
Gráficas de la primera y última configuracion de spines, pero al haber añadido el slider
son innecesarias.


cmap2 = plt.cm.gray
plt.figure()
plt.imshow(M[0,:,:],cmap = cmap2)
plt.title('Primera configuración de spines')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


plt.figure()
plt.imshow(M[-1,:,:],cmap = cmap2)
plt.title('Última configuración de spines')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
'''
########################################################################################    
'''
#El calor específico viene definido por la siguiente relación. C_v = (1/kT)**2 *(<E**2>-<E>**2)
#Para estos casos, el programa es muy lento y su ejecución es muy lenta. Tarda unos seis o siete 
#minutos ya que se calcula para muchas temperaturas un total de 10000 iteraciones.

def get_spin_energy(lattice, BJs):
    ms = np.zeros(len(BJs))
    E_means = np.zeros(len(BJs))
    E_stds = np.zeros(len(BJs))
    for i, bj in enumerate(BJs):
        spins, energies = MonteCarlo(it, 1/bj)[3],MonteCarlo(it, 1/bj)[1]
        ms[i] = np.mean(spins[int(it*0.9):])/(N+2)**2 #Dividimos por (N+2)^2 para normalizar.
        E_means[i] = np.mean(energies[int(it*0.9):]) #Tomamos el último 10% de los puntos para 
        E_stds[i] = np.std(energies[int(it*0.9):])   #asegurar que el sistema está en equilibrio
    return ms, E_means, E_stds
    
#BJs = np.arange(0.1, 2, 0.05)
#BJs = np.linspace(0.1,2,50)
Bjs = [0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.3,0.4,0.5,1,2,3,4,5,6,7]
BJs = np.array(Bjs)
ms_n, E_means_n, E_stds_n = get_spin_energy(Ising_1(N), BJs)

plt.figure(figsize=(8,5))
plt.plot(1/BJs, ms_n, 'o-', label = 'Magnetización en función de la temperatura')
plt.xlabel('T [J/$K_b$]')
plt.ylabel('$m$')
plt.legend(facecolor='white', framealpha=1)
plt.show()

#Con esta función el calor específico no sale adecuadamente. No sé la razón, pero el caso es que
#la magnetización si que sale correctamente.

plt.figure()
plt.plot(1/BJs, E_stds_n*BJs,label = 'Calor específico en función de la temperatura')
plt.xlabel('T [J/$K_b$]')
plt.ylabel('$C_V K_b^2$')
plt.legend()
plt.show()
'''