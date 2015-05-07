import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# simulation constants
dt = 0.025
dx = .5
L = 60.0
k = 5.
n = np.floor(L/dx)
N = n**2

# two constants to simplify the expressions
c1 = 1.j * dt / (4. * dx**2)    # i*hbar*dt/(4*m*dx^2)
c2 = 1.j * dt / 2.

# generate coordinates and initial wave-function
x, y = np.tile(np.linspace(0,L,n), n), np.repeat(np.linspace(0,L,n), n)
xGrid , yGrid = np.meshgrid(np.linspace(0,L,n),np.linspace(0,L,n))  # plotting coordinate grids

# psi = np.exp(-0.1*((x-L/4)**2)) * np.exp(1.j*k*x)  # gaussian front
psi = np.exp(-0.1*((x-L/4)**2 + (y-L/2)**2)) * np.exp(1.j*k*x)  # gaussian peak
psi /= np.sqrt(sum(abs(psi)**2*dx**2))  # normalize the wave function

# set a barrier
v = 12.5*(np.abs(L/2-x) < 0.56)

# numerically solve for the matrix equation A*psi(t+dt) = B*psi(t) where A and B
A = sparse.diags(1.+4.*c1+c2*v, 0) \
    - c1*sparse.eye(N,N,1) - c1*sparse.eye(N,N,-1) - c1*sparse.eye(N,N,n) - c1*sparse.eye(N,N,-n)
B = sparse.diags(1.-4.*c1-c2*v, 0) \
    + c1*sparse.eye(N,N,1) + c1*sparse.eye(N,N,-1) + c1*sparse.eye(N,N,n) + c1*sparse.eye(N,N,-n)

# initialize figure for plotting
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(0,2.*np.amax(np.abs(psi)**2))

for i in range(1000):

    if i%10 == 0:
        psiPlot = np.abs(psi.reshape(n,n)**2)
        wireFrame = ax.plot_wireframe(xGrid, yGrid, psiPlot, rstride = 2, cstride = 2)
        plt.pause(.001)
        ax.collections.remove(wireFrame)

    [psi, garbage] = linalg.bicgstab(A, B*psi)
