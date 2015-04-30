import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

n = 4
N = n**2
dt = 2.0
L = 4.0
dx = L/n
s = 1.j * dt / (4. * dx**2)    # i*hbar*dt/(4*m*dx^2)
# c2 = 1.j * dt / 2.

A = (1.+4.*s)*sparse.identity(N) - s*sparse.eye(N,N,1) - s*sparse.eye(N,N,-1) - s*sparse.eye(N,N,n) - s*sparse.eye(N,N,-n)
B = (1.-4.*s)*sparse.identity(N) + s*sparse.eye(N,N,1) + s*sparse.eye(N,N,-1) + s*sparse.eye(N,N,n) + s*sparse.eye(N,N,-n)

psi = np.zeros(N)
psi[round(N/2 + n/2)] = 1.0

[psi, garbage] = linalg.bicgstab(A, B*psi)
psi = psi.reshape(n,n)

plt.figure()
plt.pcolor(abs(psi)**2)
plt.colorbar()
plt.show()