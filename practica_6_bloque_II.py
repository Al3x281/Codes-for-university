# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:34:16 2023

@author: aleja
"""

import numpy as np
import matplotlib.pyplot as plt


N = 1000
hbar = 1
m = 1
L = 1
kt = 10

def E(nx,ny,nz):
    return (np.pi*hbar)/(2*m*L) * (nx**2 + ny**2 + nz**2)

pasos = 10000
def MonteCarlo(N,pasos,kt):
    energy = np.zeros(pasos)
    energy[0] = N*E(1,1,1)
    nums = np.ones((N,3), int)
    for i in range(1,pasos):
        atom = np.random.randint(N)
        dire = np.random.randint(3)
        state = np.random.randint(2)
        if state == 0:
            deltaE = (np.pi*hbar)/(2*m*L) * (2*nums[atom,dire] + 1)
            p = np.exp(-deltaE/kt)
            pr = np.random.rand()
            if p>pr and nums[atom,dire] > 1:
                nums[atom,dire] += -1
                energy[i] =  energy[i-1] + deltaE
            else:
                energy[i] = energy[i-1]
                
        elif state == 1:
            deltaE = (np.pi*hbar)/(2*m*L) * (2*nums[atom,dire] + 1)
            p = np.exp(-deltaE/kt)
            pr = np.random.rand()
            if p>pr:
                nums[atom,dire] += -1
                energy[i] =  energy[i-1] + deltaE
            else:
                energy[i] = energy[i-1]
    return energy
         
energy = MonteCarlo(N, pasos, kt)
    
plt.figure()
plt.plot(energy)
plt.show()



