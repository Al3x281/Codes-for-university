import matplotlib.pyplot as plt
import numpy as np
import time
import numba #Si no se tiene instalada la librería numba, simplemente comentar o borrar la linea para que funcione
import scienceplots #Lo mismo, si no se tiene instalada scienciplots, comentar o borrar esta linea
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import block_diag

plt.style.use(['science', 'notebook', 'grid'])


M,N = 5,5
alpha = 1
h = 1/M


def Laplace_dirichlet(M,N):
    #Creamos la matriz que representa al operador diferenciación
    diag = 2*(1+alpha)*np.ones(M)
    off = -1*np.ones(M-1)
    B = sparse.diags([diag, off, off], [0,-1,1],shape=(M,M)).toarray()
    B[0,0],B[0,1] = 1,0
    B[M-1,M-1],B[M-1,M-2] = 1,0
    Identity2 = sparse.diags([np.zeros(N), -np.ones(N), -np.ones(N)], [0,-1,1],shape=(N,N)).toarray()
    Identity2[0,1], Identity2[N-1,N-2] = 0,0
    Identity = sparse.eye(N).toarray()
    S = np.identity(M)
    S[0,0]  = 0
    S[M-1,M-1] = 0
    A1 = sparse.kron(Identity,B).toarray() + sparse.kron(Identity2,S).toarray()
    A1[0:M,0:M] = np.identity(M)
    A1[M*N-M:M*N,M*N-M:M*N] = np.identity(M)
    u0 = np.zeros(M*N)
    u0[:M] = 100
    x = spsolve(A1,u0)
    
    sol = x[:].reshape(N,M)
    return sol

M,N = 50,50
sol = Laplace_dirichlet(M, N)

plt.figure()
plt.contourf(sol[:,:], cmap = 'inferno')
C = plt.contour(sol[:,:], 8, colors='black', linewidth=.5)
plt.clabel(C, inline=1, fontsize=10)
plt.xlabel('Y')
plt.ylabel('X')
plt.show()

plt.figure()
plt.imshow(sol, cmap = 'inferno')
plt.colorbar()
plt.xlabel('Y')
plt.ylabel('X')
plt.show()


####################################################################################################
#Jacobi

#Implementación del método de jacobi mediante la fórmula vista en clase

@numba.jit
def Jacobi(M,tol):
    M0 = np.copy(M)
    cont = 0
    while True:
        M0[1:-1,1:-1] = 1/4 * (M[:-2,1:-1] + M[2:,1:-1] + M[1:-1,:-2] + M[1:-1,2:])
        M, M0 = M0,M
        cont += 1
        if np.max(abs(M-M0)) < tol:
            break
    
    return M,cont

#Creamos la matriz de ceros y añadimos la condición de contorno para la primera fila
M = np.zeros((50,50))
M[0,:] = 100

s,cont = Jacobi(M,10e-6)

#Graficamos la solución
plt.figure()
plt.imshow(s, cmap = 'inferno')
plt.colorbar()
plt.xlabel('Y')
plt.ylabel('X')
plt.title('Método de Jacobi')
plt.show()

######################################################################################################

#Condiciones de Neumann

def Laplace_neumann(M,N):
    diag = 2*(1+alpha)*np.ones(M)
    off = -1*np.ones(M-1)
    B = sparse.diags([diag, off, off], [0,-1,1],shape=(M,M)).toarray()
    B[0,0],B[0,1] = 2,0
    B[M-1,M-1],B[M-1,M-2] = 2,0
    Identity2 = sparse.diags([np.zeros(N), -np.ones(N), -np.ones(N)], [0,-1,1],shape=(N,N)).toarray()
    Identity2[0,1], Identity2[N-1,N-2] = 0,0
    Identity = sparse.eye(N).toarray()
    S = np.identity(M)
    S[0,0]  = 1
    S[M-1,M-1] = 1
    A1 = sparse.kron(Identity,B).toarray() + sparse.kron(Identity2,S).toarray()
    
    C = np.identity(M)
    C[0,0],C[0,1] = 1,0
    C[M-1,M-1],C[M-1,M-2] = 1,0
    
    A1[0:M,0:M] = C
    A1[M*N-M:M*N,M*N-M:M*N] = C

    u0 = np.zeros(M*N)
    u0[:M] = 100
    x = spsolve(A1,u0)
    
    sol = x[:].reshape(N,M)
    return sol

M,N = 50,50
sol2 = Laplace_neumann(M, N)

plt.figure()
plt.contourf(sol2[:,:], cmap = 'inferno')
C = plt.contour(sol2[:,:], 8, colors='black', linewidth=.5)
plt.clabel(C, inline=1, fontsize=10)
plt.xlabel('Y')
plt.ylabel('X')
plt.title('Condiciones de Von Neumann')
plt.show()

plt.figure()
plt.imshow(sol2, cmap = 'inferno')
plt.colorbar()
plt.xlabel('Y')
plt.ylabel('X')
plt.title('Condiciones de Von Neumann')
plt.show()


######################################################################################################
