import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
import time
import numba #Si no se tiene instalada la librería numba, simplemente comentar o borrar la linea para que funcione
import scienceplots #Lo mismo, si no se tiene instalada scienciplots, comentar o borrar esta linea
from scipy.optimize import minimize
from scipy.optimize import curve_fit

plt.style.use(['science', 'notebook', 'grid'])

N = 25 #Tamaño de la matriz de la que vamos a resolver el sistema

#Se inicializan las condiciones del sistema, en este caso, de manera aleatoria
d = np.random.random(N)
o = np.random.random(N-1) 
u = np.random.random(N-1)
r = np.random.random(N)

@numba.jit
def TridiagonalSolver(d,o,u,r):
    #Creamos los vectores auxiliares que necesitamos para resolver el algoritmo
    h = np.zeros(len(o))
    p = np.zeros(len(r))
    x = np.zeros(len(r))
    h[0] = o[0]/d[0]
    p[0] = r[0]/d[0]
    for i in range(1,len(o)):
        h[i] = o[i]/(d[i]- (u[i-1]*h[i-1]))
        p[i] = (r[i] - (u[i-1]*p[i-1]))/(d[i] - (u[i-1]*h[i-1]))
    p[-1] = (r[-1] - u[-1]*p[-2])/(d[-1] - u[-1]*h[-1])
    
    x[-1] = p[-1]
    #Una vez calculados los vectores auxiliares, resolvemos las x, yendo en sentido inverso
    for i in range(len(x)-2,-1,-1):
        x[i] = p[i] - h[i]*x[i+1]
    return x


#Comprobamos que la solución es correcta con otros métodos implementados en numpy
x1 = TridiagonalSolver(d,o,u,r)
print(x1)     

tridiagonal_matrix = diags([u, d, o], offsets=[-1, 0, 1], shape=(N, N))
A = tridiagonal_matrix.toarray()  

x2 = np.linalg.solve(A,r)
print(x2) 

x3 = np.dot(np.linalg.inv(A),r)
print(x3) 


##################################################################################################
#Gráfica para el algoritmo de Thomas


#Se resuelve el algoritmo de Thomas para un número creciente de números y se calcula la diferencia
#entre el tiempo de inicio y de fin para calcular la duración de la ejecución

@numba.jit #Si no se tiene instalada la librería numba, simplemente comentar o borrar la linea para que funcione
def bucletemporal(x):
    t1 = []
    for i in x:
        d = np.random.random(i)
        o = np.random.random(i-1)
        u = np.random.random(i-1)
        r = np.random.random(i)
        start_time1 = time.perf_counter_ns()
        x = TridiagonalSolver(d,o,u,r)
        end_time1 = time.perf_counter_ns()
        t1.append((end_time1 - start_time1))
    return t1

x = np.arange(1,100000,99,dtype= int)
t1 = bucletemporal(x)

plt.figure()
plt.plot(x,t1,'o' ,color = 'red', label = 'Algoritmo de Thomas')
plt.xlabel('Dimensión de matriz')
plt.ylabel('Tiempo de ejecución (ns)')
plt.title('Ejecución para Thomas')
plt.show()


###################################################################################################
#Gráfica de las tres al mismo tiempo

#Realizamos el mismo procedimiento pero para los tres a la vez con un número menor de puntos
#que en el caso anterior ya que es más complicada la ejecución de algoritmos como el cálculo de la inversa

@numba.jit #Si no se tiene instalada la librería numba, simplemente comentar o borrar la linea para que funcione
def bucletemp(N):
    t1 = []
    t2 = []
    t3 = []
    y = np.arange(1,N,49,dtype= int)
    for i in y:
        d = np.random.random(i)
        o = np.random.random(i-1)
        u = np.random.random(i-1)
        r = np.random.random(i)
        
        tridiagonal_matrix = diags([u, d, o], offsets=[-1, 0, 1], shape=(i, i))
        A = tridiagonal_matrix.toarray()
        
        start_time1 = time.perf_counter_ns()
        x = TridiagonalSolver(d,o,u,r)
        end_time1 = time.perf_counter_ns()
        t1.append((end_time1 - start_time1))
        
        start_time2 = time.perf_counter_ns()
        x = np.dot(np.linalg.inv(A),r)
        end_time2 = time.perf_counter_ns()
        t2.append((end_time2 - start_time2))
    
        start_time3 = time.perf_counter_ns()
        x = np.linalg.solve(A, r)
        end_time3 = time.perf_counter_ns()
        t3.append((end_time3 - start_time3))
    return t1,t2,t3


n = 2000
t1,t2,t3 = bucletemp(n)
y = np.arange(1,n,49,dtype= int)

#Graficamos los resultados para los tiempos de ejecución de los diferentes algoritmos
plt.figure()
plt.plot(y,t1,'o' ,color = 'red', label = 'Algoritmo de Thomas')
plt.plot(y,t2,'o', color = 'blue', label = 'Matriz inversa')
plt.plot(y,t3, 'o', color = 'green', label = 'np.linalg.solve()')
plt.legend(loc = 'best', fontsize='large')
plt.xlabel('Dimensión de matriz')
plt.ylabel('Tiempo de ejecución')
plt.title('Comparativa entre algoritmos')
plt.show()
####################################################################################################
