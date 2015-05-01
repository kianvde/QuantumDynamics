import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# simulation constants
dt = 0.025
dx = 1.
L = 60.0
k = 5.
n = np.floor(L/dx)
N = n**2

# two constants to simplify the expressions
c1 = 1.j * dt / (4. * dx**2)    # i*hbar*dt/(4*m*dx^2)
# c2 = 1.j * dt / 2.

# # numerically solve for the matrix equation A*psi(t+dt) = B*psi(t) where A and B
A = (1.+4.*c1)*sparse.identity(N) \
    - c1*sparse.eye(N,N,1) - c1*sparse.eye(N,N,-1) - c1*sparse.eye(N,N,n) - c1*sparse.eye(N,N,-n)
B = (1.-4.*c1)*sparse.identity(N) \
    + c1*sparse.eye(N,N,1) + c1*sparse.eye(N,N,-1) + c1*sparse.eye(N,N,n) + c1*sparse.eye(N,N,-n)

# generate x-coordinates and initial wave-function
x = np.linspace(0,L,n)
xp , yp = np.meshgrid(x,x)  # plotting coordinate grids
# psi = np.exp(-0.1*((np.tile(x, n)-L/4)**2 + (np.repeat(x, n)-L/2)**2)) * np.exp(1.j*k*np.tile(x, n))  # gaussian peak
psi = np.exp(-0.1*((np.tile(x, n)-L/4)**2)) * np.exp(1.j*k*np.tile(x, n))  # gaussian front
psi /= np.sqrt(sum(abs(psi)**2*dx**2))  # normalize the wave function

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ims = []
# np.set_printoptions(precision=2,threshold=np.nan)
for _ in range(1):
    [psi, garbage] = linalg.bicgstab(A, B*psi)
    psiPlot = np.abs(psi.reshape(n,n))**2

    # print to check if psiPlot is constant along the first dimension
    print np.amax(abs(np.abs(psiPlot - psiPlot[30, :])))

    im = ax.plot_surface(xp, yp, psiPlot),
    ims.append(im)

# problems with animation; should be a gaussian wave packet along the x direction and constant
# in the y direction.

# the array np.abs(psi.reshape(n,n))**2 is correct though, see print
try: input("continue to animation")
except SyntaxError: pass
im_ani = animation.ArtistAnimation(fig2, ims, interval=1000, repeat_delay=3000, blit=True)
plt.show()
