import matplotlib.pyplot as plt
import numpy as np

Max = 1
N = 10000


def f(x): 
    return np.sin(1/(x*(2-x)))**2



def montecarlo(f,Max,N):
    p = np.random.uniform(0,Max,N)
    x = np.random.uniform(0,2,N)
    cond = p<f(x)
    s = np.sum(cond)/N * Max *2
    err = np.sqrt((s*(Max*2 - s))/N)
    return s,err,p,x,cond

s,err,p,x,cond = montecarlo(f,Max, N)

x1 = np.linspace(0.001,1.999,N)
plt.figure()
plt.plot(x1,f(x1))
plt.scatter(x[cond],p[cond])
plt.show()


pasos = np.arange(1e3,1e5,1e3)
s1 = np.zeros(len(pasos))
error1 = np.zeros(len(pasos))

for i in range(len(pasos)):
    s1[i] = montecarlo(f,Max, int(pasos[i]))[0]
    error1[i] = montecarlo(f,Max, int(pasos[i]))[1]
    

fig,ax = plt.subplots()
ax.plot(pasos,s1,'o')
ax.axhline(1.4514,0,1)
plt.show()

plt.figure()
plt.plot(pasos,error1)
plt.show()



def g(x):
    return np.exp(-x**2)


s,err,p,x,cond = montecarlo(g,Max, N)

x1 = np.linspace(0.001,1.999,N)
plt.figure()
plt.plot(x1,g(x1))
plt.scatter(x[cond],p[cond], s= 1, color = 'r')
plt.show()

# =============================================================================
# Esfera de diez dimensiones
# =============================================================================

w = np.random.uniform(-1,1,(N,10))
r = np.zeros(N)

for i in range(N):
    r[i] = np.linalg.norm(w[i])**2

cond1 = r <= 1

s1 = np.sum(cond)/N  
    
    