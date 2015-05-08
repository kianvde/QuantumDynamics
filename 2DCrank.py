import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc

# simulation constants
dt = 0.05
dx = .5
L = 60.0
k = 15.
n = np.floor(L/dx)
N = n**2

# two constants to simplify the expressions
c1 = 1.j * dt / (4. * dx**2)    # i*hbar*dt/(4*m*dx^2)
c2 = 1.j * dt / 2.

# generate coordinates and initial wave-function
x, y = np.tile(np.linspace(0,L,n), n), np.repeat(np.linspace(0,L,n), n)
xGrid , yGrid = np.meshgrid(np.linspace(0,L,n),np.linspace(0,L,n))  # plotting coordinate grids

psi = np.exp(-0.1*((x-L/4)**2)) * np.exp(1.j*k*x)  # gaussian front
# psi = np.exp(-0.05*((x-L/4)**2 + (y-L/2)**2)) * np.exp(1.j*k*x)  # gaussian peak
psi /= np.sqrt(sum(abs(psi)**2*dx**2))  # normalize the wave function

# set a barrier
v = 12.5*(np.abs(3.*L/8.-x) < 0.56)
v2 = 1.0*((np.abs(7.*L/16.-y) > 0.6))
v3 = 1.0*((np.abs(9.*L/16.-y) > 0.6))
v = v2 * v * v3

# numerically solve for the matrix equation A*psi(t+dt) = B*psi(t) where A and B
Adiag = np.ones((1,n))
Adiag[0,n-1] = 0.0
Adiag = np.tile(Adiag, n)
A = sparse.diags(1.+4.*c1+c2*v, 0) \
    - c1*sparse.eye(N,N,n) - c1*sparse.eye(N,N,-n) \
    - c1*sparse.diags(Adiag[0,0:(N-1)],-1) - c1*sparse.diags(Adiag[0,0:(N-1)],1)
B = sparse.diags(1.-4.*c1-c2*v, 0) \
     + c1*sparse.eye(N,N,n) + c1*sparse.eye(N,N,-n) \
     + c1*sparse.diags(Adiag[0,0:(N-1)],-1) + c1*sparse.diags(Adiag[0,0:(N-1)],1)

# initialize figure for plotting
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(0,2.*np.amax(np.abs(psi)**2))

crit = 0.
psiPlot = np.abs(psi.reshape(n,n)**2)
Inter = psiPlot[:,np.round(3.*n/4.)] # Screen set at x = 0.75*L
for i in range(500):

    if i%10 == 0:
        psiPlot = np.abs(psi.reshape(n,n)**2)
        wireFrame = ax.plot_wireframe(xGrid, yGrid, psiPlot, rstride = 2, cstride = 2)
        if sum(psiPlot[:,np.round(3.*n/4.)]) > crit:
            Inter = psiPlot[:,np.round(3.*n/4.)]
            crit = sum(psiPlot[:,np.round(3.*n/4.)])

        plt.pause(.001)
        ax.collections.remove(wireFrame)

    [psi, garbage] = linalg.bicgstab(A, B*psi,x0=psi)

plt.figure()
plt.plot(np.linspace(0,L,n),Inter,'b')
plt.title("Quantum double slit")
plt.xlabel("x")
rc('text', usetex=True)
plt.ylabel("$|\psi|^2$")
plt.show()
