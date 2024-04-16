import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import scienceplots

plt.style.use(['science', 'notebook', 'grid'])

def density(K,P):
    gamma = 4/3 
    return ( P / K )**( 1 / gamma )

#Derivative of energy for the pp-chain
def energy_pp(rho,T,X):
    T6 = T/1e6
    L_M = 1.96 #Relation between solar luminosity and solar mass 
    M_R3 = 5.9 #Relation between solar mass and radius 
    return 3e6 * X**2 * rho * T6**(-2/3) * np.exp(-33.8/T6**(1/3)) * M_R3 / L_M 

    
# Definition of the differential equations
def equation(y,x):
    r, m, L, logT = y
    P = np.exp(x)  
    T = np.exp(logT)
    K = 0.37
    X = 1
    kap = 0.145
    drdx = - ( r**2 * P ) / ( m * density(K,P))
    dmdx = - (4 * np.pi * r**4 * P) / m
    #dlogTdx = 1/4
    dlogTdx = (4*np.pi*kap*L*P)/(m*T**4)
    dLdx = -(r**4 * P * energy_pp(density(K,P),T,X)) / m
    return [drdx, dmdx, dLdx, dlogTdx]

# Solving points
Pmin = 1e-10
P0 = 23.6
x = np.linspace(np.log(Pmin) , np.log(P0), 100)[::-1]
Pnorm = np.linspace(P0,Pmin,100)

#Initial conditions

r0 = 1e-4
m0 =  4/3 * np.pi * r0**3 * density(0.37, P0)
T0 = 15e6
L0 = m0 * energy_pp(density(0.37, P0),T0,1)
y0 = [r0,m0,L0,np.log(T0)]


#Con ODEINT
# Solution of the four differential equations
solution = odeint(equation, y0, x)

# Extraemos las soluciones
r_sol = solution[:, 0]
m_sol = solution[:, 1]
L_sol = solution[:, 2]
logT_sol = solution[:, 3]


'''
#Con solve_ivp
solution = solve_ivp(equation,[np.log(P0), np.log(Pmin)],y0, t_eval = x, method = 'LSODA')
r_sol = solution.y[0,:]
m_sol = solution.y[1,:]
L_sol = solution.y[2,:]
logT_sol = solution.y[3,:]
'''
# =============================================================================
# Graphics in normalized units (Inverting X-axis)
# =============================================================================

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

 # Plot for radius
axs[0, 0].plot(x, r_sol)
axs[0, 0].invert_xaxis()
axs[0, 0].set_xlabel('log(P)')
axs[0, 0].set_ylabel('Radius $(\\frac{R}{R_{\\odot}})$')
axs[0, 0].set_title('Solar Radius')

# Plot for mass
axs[0, 1].plot(x, m_sol)
axs[0, 1].invert_xaxis()
axs[0, 1].set_xlabel('log(P)')
axs[0, 1].set_ylabel('Mass $(\\frac{M}{M_{\\odot}})$')
axs[0, 1].set_title('Solar Mass')

# Plot for luminosity
axs[1, 0].plot(x, L_sol)
axs[1, 0].invert_xaxis()
axs[1, 0].set_xlabel('log(P)')
axs[1, 0].set_ylabel('$Luminosity (\\frac{L}{L_{\\odot}})$')
axs[1, 0].set_title('Solar Luminosity')

# Plot for temperature
axs[1, 1].plot(x, logT_sol)
axs[1, 1].invert_xaxis()
axs[1, 1].set_xlabel('log(P)')
axs[1, 1].set_ylabel('log(T)')
axs[1, 1].set_title('Solar Temperature (Fully convective)')

plt.suptitle('SOLAR PROPERTIES WITH DIMENSIONLESS UNITS (INVERTED X-AXIS)', fontsize=16)

# Adjust layout
plt.tight_layout()

#Show graphics
plt.show()

# =============================================================================
# Graphics in normalized units
# =============================================================================

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot for radius
axs[0, 0].plot(x, r_sol)
axs[0, 0].set_xlabel('log(P)')
axs[0, 0].set_ylabel('Radius $(\\frac{R}{R_{\\odot}})$')
axs[0, 0].set_title('Solar Radius')

# Plot for mass
axs[0, 1].plot(x, m_sol)
axs[0, 1].set_xlabel('log(P)')
axs[0, 1].set_ylabel('Mass $(\\frac{M}{M_{\\odot}})$')
axs[0, 1].set_title('Solar Mass')

# Plot for luminosity
axs[1, 0].plot(x, L_sol)
axs[1, 0].set_xlabel('log(P)')
axs[1, 0].set_ylabel('$Luminosity (\\frac{L}{L_{\\odot}})$')
axs[1, 0].set_title('Solar Luminosity')

# Plot for temperature
axs[1, 1].plot(x, logT_sol)
axs[1, 1].set_xlabel('log(P)')
axs[1, 1].set_ylabel('log(T)')
axs[1, 1].set_title('Solar Temperature (Fully convective)')

plt.suptitle('SOLAR PROPERTIES WITH DIMENSIONLESS UNITS', fontsize=16)

# Adjust layout
plt.tight_layout()

#Show graphics
plt.show()

# =============================================================================
# Graphics for solar properties with units (cgs)
# =============================================================================

#Units for the solar properties
prop = [1.988e33, 6.96e10, 3.99e33, 1.12e16]

Mfac, Rfac, Lfac, Pfac = prop

#Solutions with units
R_solar = r_sol * Rfac
M_solar = m_sol * Mfac
L_solar = L_sol * Lfac
T_solar = np.exp(logT_sol)
P_solar = Pnorm * Pfac

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot for radius
axs[0, 0].plot(P_solar, R_solar)
axs[0, 0].set_xlabel('P(c.g.s)')
axs[0, 0].set_ylabel('Radius $(cm)$')
axs[0, 0].set_title('Solar Radius')

# Plot for mass
axs[0, 1].plot(P_solar, M_solar)
axs[0, 1].set_xlabel('P(c.g.s)')
axs[0, 1].set_ylabel('Mass $(g)$')
axs[0, 1].set_title('Solar Mass')

# Plot for luminosity
axs[1, 0].plot(P_solar, L_solar)
axs[1, 0].set_xlabel('P(c.g.s)')
axs[1, 0].set_ylabel('Luminosity $(\\frac{erg}{s})$')
axs[1, 0].set_title('Solar Luminosity')

# Plot for temperature
axs[1, 1].plot(P_solar, T_solar)
axs[1, 1].set_xlabel('P(c.g.s)')
axs[1, 1].set_ylabel('T(K)')
axs[1, 1].set_title('Solar Temperature (Fully convective)')

plt.suptitle('SOLAR PROPERTIES WITH UNITS', fontsize=16)

# Adjust layout
plt.tight_layout()

#Show graphics
plt.show()


# =============================================================================
# Graphics of the radius and the mass for different initial pressure conditions
# =============================================================================

#Modification of density 
def density1(K,P):
    gamma = 4/3 + 0.001
    return ( P / K )**( 1 / gamma )

# Modification of the function to make it shorter and specific for the study of 
# the radius and mass for different pressures at center
def equation(y, x):
    r, m = y
    P = np.exp(x)  # because x = log(P)
    K = 0.37
    drdx = - ( r**2 * P ) / ( m * density1(K,P))
    dmdx = - (4 * np.pi * r**4 * P) / m
    return [drdx, dmdx]

Ps = np.linspace(1,32)
y1 = [r0,m0]
ms = []
rs = []

for i in Ps:
    Pmin = 1e-10
    x = np.linspace(np.log(Pmin) , np.log(i), 100)[::-1]
    Pnorm = np.linspace(P0,Pmin,100)
    solution = odeint(equation, y1, x)
    rs.append( solution[-1, 0])
    ms.append( solution[-1, 1])

rs = np.array(rs)
ms = np.array(ms)

rs_units = rs * Rfac
ms_units = ms * Mfac
Ps_units = Ps * Pfac
    
# Create a 1x2 grid of subplots
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Plot for radius
sc = ax[0].scatter(rs, ms,c = Ps, cmap = 'hot')
ax[0].set_ylabel('Mass $(\\frac{M}{M_{\\odot}})$')
ax[0].set_xlabel('Radius$(\\frac{R}{R_{\\odot}})$')
ax[0].set_title('Unitless plot')

# Add colorbar to the first subplot
fig.colorbar(sc, ax=ax[0], label='Pressure $(\\frac{GM_{\\odot}^{2}}{R_{\\odot}^{4}})$')

sc = ax[1].scatter(rs_units, ms_units ,c = Ps_units , cmap = 'hot')
ax[1].set_ylabel('Mass (g)')
ax[1].set_xlabel('Radius $(cm)$')
ax[1].set_title('Plot with units')

# Add colorbar to the second subplot
fig.colorbar(sc, ax=ax[1], label='Pressure(c.g.s)')


# Adjust layout
plt.tight_layout()
plt.show()


# =============================================================================
# Modelo de densidad lineal 
# =============================================================================

def lin_dens(r, rho_c):
    return (1-r/Rfac)*rho_c

# Definition of the differential equations
def equation_lin(y, x):
    r, m, L, logT = y
    P = np.exp(x)  
    T = np.exp(logT)
    X = 1
    rho_c = 15
    drdx = - ( r**2 * P ) / ( m * lin_dens(r,rho_c))
    dmdx = - (4 * np.pi * r**4 * P) / m
    dlogTdx = 1/4
    dLdx = -(r**4 * P * energy_pp(lin_dens(r,rho_c),T,X)) / m
    return [drdx, dmdx, dLdx, dlogTdx]

# Solving points
Pmin = 1e-2
P0 = 10
x = np.linspace(np.log(Pmin) , np.log(P0), 100)[::-1]
Pnorm = np.linspace(P0,Pmin,100)

#Initial conditions

r0 = 1e-4
rho0 = 50 * 5.9
m0 =  4/3 * np.pi * r0**3 * lin_dens(r0,rho0)
T0 = 10e6
L0 = m0 * energy_pp(lin_dens(r0, rho0),T0,1)
y0 = [r0,m0,L0,np.log(T0)]

# Solution of the four differential equations
solution_lin = odeint(equation_lin, y0, x)

# Extraemos las soluciones
r_lin = solution_lin[:, 0]
m_lin = solution_lin[:, 1]
L_lin = solution_lin[:, 2]
logT_lin = solution_lin[:, 3]    


# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot for radius
axs[0, 0].plot(x, r_lin)
axs[0, 0].invert_xaxis()
axs[0, 0].set_xlabel('log(P)')
axs[0, 0].set_ylabel('Radius $(\\frac{R}{R_{\\odot}})$')
axs[0, 0].set_title('Radius')

# Plot for mass
axs[0, 1].plot(x, m_lin)
axs[0, 1].invert_xaxis()
axs[0, 1].set_xlabel('log(P)')
axs[0, 1].set_ylabel('Mass $(\\frac{M}{M_{\\odot}})$')
axs[0, 1].set_title('Mass')

# Plot for luminosity
axs[1, 0].plot(x, L_lin)
axs[1, 0].invert_xaxis()
axs[1, 0].set_xlabel('log(P)')
axs[1, 0].set_ylabel('$Luminosity (\\frac{L}{L_{\\odot}})$')
axs[1, 0].set_title('Luminosity')

# Plot for temperature
axs[1, 1].plot(x, logT_lin)
axs[1, 1].invert_xaxis()
axs[1, 1].set_xlabel('log(P)')
axs[1, 1].set_ylabel('log(T)')
axs[1, 1].set_title('Temperature (Fully convective)')

plt.suptitle('PROPERTIES WITH DIMENSIONLESS UNITS (INVERTED X-AXIS) \n $P_o = {}$ ; $\\rho_o = {}$ ; $X = 1$ ; $T_o = {}$'.format(P0,rho0,T0), fontsize=16)

# Adjust layout
plt.tight_layout()

#Show graphics
plt.show()





