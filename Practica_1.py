import numpy as np

import sympy
from sympy import *


import sympy as sp
sp.init_printing(use_latex='mathjax')

#from sympy.mpmath import *
from gravipy import *

#from sympy import init_printing
#init_printing() # doctest: +SKIP


t,x,y,z,Omega, Gamma=symbols('t x y z Omega Gamma')
#x,y,Gamma = symbols('x y Gamma')
#theta, phi, Gamma =symbols('theta, phi, Gamma')
#t,r, theta, phi, Gamma , Lambda, Phi = symbols('t,r, theta, phi, Gamma, Lambda, Phi')
#t,x, y, z, Gamma, ft, fz = symbols('t,x,y,z,Gamma, ft, fz')
#t,x, y, z, Gamma,c,omega,phip,phic,A= symbols('t,x,y,z,Gamma,c,omega,phip,phic,A')
#r, theta, Gamma = symbols('r, theta, Gamma')
#M=symbols('M')
#a=symbols('a')
#a = sympy.Function('a')(r)
#b = sympy.Function('b')(r)
xi = Coordinates('chi', (t,x,y,z))
#xi = Coordinates('chi', (r,theta))
#xi = Coordinates('chi', (theta,phi))
#xi = Coordinates('chi', (x,y))
#a = sympy.Function('a')(r)
#b= sympy.Function('b')(r)
#ft= sympy.Function('ft')(z)
#fz= sympy.Function('fz')(z)

nd=4
ndp1=nd+1

#Integral(t,t)
xi(-1)
for i in range(1,ndp1):
    sp.pprint(xi(-i))
#Metric = diag(1,sin(theta)**2)  
#Metric = diag(1/(1-r),r**2)   
#Metric = diag(-ft,1,1,fz) 
#Metric = diag(1/y**2, 1/y**2)
#Metric = diag(1, r**2)
#Metric = diag(-(1-2*M/r), 1/(1-2*M/r), r**2, r**2*sin(theta)**2)
#Metric = diag(-exp(2*a), exp(2*b), r**2, r**2*sin(theta)**2)
#Schwarzschild in isotropic coords
#Metric = diag(-((1-M/(2*r))/(1+M/(2*r)))**2, (1+M/(2*r))**4,(1+M/(2*r))**4*r**2, (1+M/(2*r))**4*r**2*sin(theta)**2)
#Rotating
Metric = Matrix([[-1+(x**2+y**2)*Omega**2,-Omega*y,Omega*x,0],[-Omega*y,1,0,0],[Omega*x,0,1,0],[0,0,0,1]])
#GW
#Metric = Matrix([[-c**2,0,0,0],[0,1+A*cos(omega*z/c-omega*t),A*cos(omega*z/c-omega*t),0],[0,A*cos(omega*z/c-omega*t),1+A*cos(omega*z/c-omega*t),0],[0,0,0,1]])
#Metric = Matrix([[-1,0,0,0],[0,1+A*cos((omega/c)*(z-t)),A*cos((omega/c)*(z-t)),0],[0,A*cos((omega/c)*(z-t)),1+A*cos((omega/c)*(z-t)),0],[0,0,0,1]])
#Kerr
#Metric = Matrix([[1,0,0,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
#rhosq = r **2+( a **2) * cos ( theta ) **2
#Delta = r **2 -2* M * r + a **2
#Metric = Matrix ([[(1 -(2* M * r ) / rhosq ) ,0 ,0 ,(2* a * M * r * sin ( theta ) **2) / rhosq ] ,[0 , -
#rhosq / Delta ,0 ,0] ,[0 ,0 , - rhosq ,0] ,[(2* a * M * r * sin ( theta ) **2) / rhosq ,0 ,0 , -( sin (
#theta ) **2) *(( r **2+ a **2) +(2*( a **2) * M * r * sin ( theta ) **2) / rhosq ) ]])
g = MetricTensor('g', xi, Metric)
for i in range(1,ndp1):
    for j in range(1,ndp1):
        print ('g('+ str(i-1)+',' + str(j-1)+')=') 
        print ("\n")
        sp.pprint(g(i,j))
        print ("\n")


#print(sympy.diff(g(1,1),xi(-1)))        
#print(g(1,1))
Ga = Christoffel('Gamma', g)
for i in range (1,ndp1):
    for j in range (1,ndp1):
        for k in range (1,ndp1):
            if Ga(-i,j,k) != 0:
                sp.pprint(Gamma,)
                print('('+ str(i-1)+',' +str(j-1)+','+ str(k-1)+')=') 
#                print('(',i-1,j-1,k-1,")=")
                sp.pprint(Ga(-i,j,k))
                pass
            
            
Ri = Ricci('Ri', g)
for i in range (1,ndp1):
    for j in range (1,ndp1):
#        if Ri(i,j) != 0:
        print('R('+ str(i-1)+',' + str(j-1)+')=') 
        print("\n")    
        sp.pprint(Ri(i,j))
        print("\n")
            
R=0
for i in range(1,ndp1):
     R=R+Ri(-i,i)
print('R=')
print(R)    
            
G = Einstein('G',Ri )

for i in range (1,ndp1):
    for j in range (1,ndp1):
#        if G(i,j) != 0:
            print('G('+ str(i-1)+',' + str(j-1)+')=') 
            print("\n")    
            sp.pprint(G(i,j))
            print("\n")
        
                
 
                
Rm = Riemann('Rm', g)
#for i in range (1,ndp1):
i=2
for j in range (1,ndp1):
    for k in range (1,ndp1):
        for l in range (1,ndp1):
            if Rm(i,j,k,l) != 0:
                print('Rm('+ str(i-1)+',' + str(j-1)+','+ str(k-1)+','+ str(l-1)+',',')=') 
                print("\n")    
                sp.pprint(Rm(-i,j,k,l))
                print("\n")
tau = symbols('tau')
w = Geodesic('w', g, tau)
#for i in range (1,ndp1):
#    print ('geod('+ str(i-1)+'):') 
#    sp.pprint(w(i))
#sp.pprint(xi(-1))
#sp.pprint('Prueba de texto = ')
#sp.pprint(xi(-2))
#ricci = sum([Rm(i, All, k, All)*g(-i, -k) for i, k in list(variations(range(1, 3), 2, True))], zeros(2))                
#ricci.simplify() 
#print(ricci)                   
