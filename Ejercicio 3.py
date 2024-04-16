import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import block_diag
from sympy import symbols, Eq, solve, cos, sin, exp



def Euler_forward_matriz(x0,v0,w,alpha,dt,t_tot):
    y = [[x0,v0]] #Condiciones iniciales
    r = np.array([[x0],[v0]])
    t = [0] #array de tiempo transcurrido en cada iteración
    s = 0 #Contador de paso de tiempo
    A = np.array([[1,dt],[-dt* w**2, 1-alpha*dt]]) #Definimos la matriz del sistema discretizado
    for i in range(int(t_tot/dt)):
        x = np.dot(A,r)
        r = np.array(x.flatten('C'))
        s += dt
        y.append(r)
        t.append(s)
    
    z = np.array(y)
    return z,t


def Euler_backward_matrix(x0, v0, w, alpha, dt, t_tot):
    y = [[x0, v0]] #Condiciones iniciales
    r = np.array([[x0], [v0]])
    t = [0] #array de tiempo transcurrido en cada iteración
    s = 0 #Contador de paso de tiempo
    #Definimos la matriz del sistema discretizado
    A = np.array([[1, -dt], [dt * w**2, 1 + alpha * dt]])
    B = np.linalg.inv(A)
    for i in range(int(t_tot / dt)):
        #x = np.linalg.solve(A, r)
        x = np.dot(B,r)
        r = np.array(x.flatten('C'))
        s += dt
        y.append(r)
        t.append(s)

    z = np.array(y)
    return z, t


def Crank_Nicholson(x0, v0, w, alpha, dt, t_tot):
    y = [[x0, v0]] #Condiciones iniciales
    r = np.array([[x0], [v0]])
    t = [0] #array de tiempo transcurrido en cada iteración
    s = 0 #Contador de paso de tiempo
    #Definimos las matrices del sistema discretizado
    A = np.array([[1, -dt/2], [(dt/2) * w**2, 1 + alpha * (dt/2)]])
    B = np.array([[1,dt/2],[-(dt/2) * w**2, 1 - alpha * (dt/2)]])
    C = np.linalg.inv(A)
    for i in range(int(t_tot / dt)):
        x = np.dot(C,np.dot(B,r))
        r = np.array(x.flatten('C'))
        s += dt
        y.append(r)
        t.append(s)

    z = np.array(y)
    return z, t

x0,v0,w,alpha,dt,t_tot = 1,0,0.5,0.2,0.7,50 #Definimos las condiciones iniciales y las constantes
x1,t1 = Euler_forward_matriz(x0, v0, w, alpha, dt,t_tot) 
x2,t2 = Euler_backward_matrix(x0, v0, w, alpha, dt, t_tot)
x3, t3 = Crank_Nicholson(x0, v0, w, alpha, dt, t_tot)


################################################################################################
#Estudio del error para los distintos métodos

#Definimos la solución analítica
def osc_amor_anal(x0,v0,w,alpha,t_tot,dt):
    t = np.arange(0,t_tot+dt,dt)
    if alpha**2 - 4*w**2 > 0:
        lam1 = (-alpha - np.sqrt(alpha**2 - 4*w**2))/2 
        lam2 = (-alpha + np.sqrt(alpha**2 - 4*w**2))/2
        #Planteamos un sistema para obtener las constantes en funcion de las condiciones inciales
        B = np.array([[1,1],[lam1,lam2]])
        C = np.linalg.inv(B)
        r = np.array([[x0], [v0]])
        A1,A2 = np.dot(C,r)
        return A1*np.exp(lam1*t) + A2*np.exp(lam2*t), t #Devolvemos la solución y el array de tiempos

    if alpha**2 == 4*w**2:
        #Planteamos un sistema para obtener las constantes en funcion de las condiciones inciales
        B = np.array([[1,0],[-alpha/2,1]])
        C = np.linalg.inv(B)
        r = np.array([[x0], [v0]])
        A1,A2 = np.dot(C,r)
        return A1*np.exp(-alpha/2 * t) + A2*t*np.exp(-alpha/2 * t),t #Devolvemos la solución y el array de tiempos
    
    if alpha**2 < 4*w**2 :
        #El sistema es no lineal, lo resolvemos analíticamente con sympy
        A, phi = symbols('A phi')
        epsilon = np.sqrt(w**2 - (alpha /2)**2)
        eq1 = Eq(A * cos(phi), x0)
        eq2 = Eq(A * (alpha/2) * cos(phi) + A * epsilon * sin(phi), v0)
        sol = solve((eq1, eq2), (A, phi))
        A_v, phi_v = sol[0]
        A_v,phi_v = float(A_v), float(phi_v)
        return A_v * np.exp(-alpha/2 * t) * np.cos(epsilon*t + phi_v),t #Devolvemos la solución y el array de tiempos

x5, t = osc_amor_anal(x0, v0, w, alpha, t_tot, dt)

#Graficamos todas las soluciones comparando con la analítica
plt.figure()
plt.plot(t1,x1[:,0], label ='Euler Forward')
plt.plot(t2,x2[:,0], label ='Euler Backward')
plt.plot(t3,x3[:,0], label ='Crank-Nicholson')
plt.plot(t,x5, label = 'Solución Analítica')
plt.legend(loc = 'best', fontsize='large')
plt.xlabel('Tiempo')
plt.ylabel('Posición')
plt.show()

#Comparamos las funciones restándolas con la analítica

plt.figure()
plt.plot(t1,np.abs(x5[0:-1] - x1[:,0]), label =' Error Euler Forward')
plt.plot(t2,np.abs(x5[0:-1] - x2[:,0]), label ='Error Euler Backward')
plt.plot(t3,np.abs(x5[0:-1] - x3[:,0]), label ='Error Crank-Nicholson')
plt.legend(loc = 'best', fontsize='large')
plt.xlabel('Tiempo')
plt.ylabel('Diferencia con Analítica')
plt.show()







