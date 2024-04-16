# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 09:04:10 2022

@author: aleja
"""

import random 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import poisson
import scipy.special

p = 0.99

def Camino_Aleatorio(N,x0,p):
    trayectoria = np.zeros(N,dtype = np.int64)
    trayectoria[0]=x0
    for i in range(1,N):
        trayectoria[i] = trayectoria[i-1] + np.random.choice([0,1], p=[p,1-p])
        
    return trayectoria

N = 500

I = 1000

x = []

for i in range(I+1):
    x.append(Camino_Aleatorio(N,0,p)[-1])
    
 
media = np.mean(x)    
sigma = np.std(x)

def normal(x,mu, sigma):
    return 2*(1/((np.sqrt(2*np.pi))*sigma))*np.exp(-((x-mu)**2)/(2*sigma**2))


def Poisson(p,N,x): 
    lamb = int((1-p)*N)
    r = []
    for i in range(len(x)):
        r.append((((lamb)**abs(x[i]))/np.math.factorial(abs(x[i]))) * np.exp(-lamb))
    return np.array(r)


s= np.linspace(min(x), max(x) + 1,100)
w = np.linspace(min(x), max(x) + 1,500,dtype = np.int64)



intervalos = range(min(x), max(x) + 1) #calculamos los extremos de los intervalos
plt.figure()
#plt.plot(s, normal(s,media,sigma),'--', color = 'k')
#plt.plot(w,Poisson(p,N,w),'--', color = 'r')
plt.hist(x=x, bins= intervalos, density = False, color='#F2AB6D', rwidth=0.85)
plt.title('Histograma de posiciones')
plt.xlabel('Posiciones')
plt.ylabel('Frecuencia')
plt.xticks(10)






