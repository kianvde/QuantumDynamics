import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

dt = 2.0
n = 100         # number of points
L = 10.0
dx = L/n
c1 = 1.j * dt / (4. * dx**2)    # i*hbar*dt/(4*m*dx^2)
c2 = 1.j * dt / 2.              # i*dt/(2*hbar)
k = 5.
bound = 0

# set a potential
v = 1.0*np.zeros(n)

## numerically solve for the matrix equation A*psi(t+dt) = B*psi(t) where A and B are tridiagonal matrices ##

A = sparse.diags(1.+2.*c1+c2*v, 0) - c1*sparse.eye(n,n,1) - c1*sparse.eye(n,n,-1)
B = sparse.diags(1.-2.*c1-c2*v, 0) + c1*sparse.eye(n,n,1) + c1*sparse.eye(n,n,-1)
if bound == 1:
    A[0,0:] = 0.
    A[n-1,0:] = 0.
    A[0,0] = 1.
    A[n-1,n-1] = 1.
    B[0,0:] = 0.
    B[n-1,0:] = 0.
    B[0,0] = 1.
    B[n-1,n-1] = 1.
else:
    A[0,n-1] = -c1
    A[n-1,0] = -c1
    B[0,n-1] = c1
    B[n-1,0] = c1


# generate x-coordinates and initial wave-function
mean = L/2
sigma = 1.0
x = np.linspace(0,L,n)
psi = np.exp(-(x-mean)**2/(2*sigma**2)) * np.exp(1.j*k*x)    # phase factor guessed
psi /= np.sqrt(sum(abs(psi)**2*dx)) # normalize the wave function
# psi[0] = 0.
# psi[n-1] = 0.

# start loop
fig2, ax = plt.subplots()
ims = []
for i in range(100):
    [psi, garbage] = linalg.bicgstab(A, B*psi)
    im, = plt.plot(x, np.abs(psi)**2, 'b')
    ims.append([im])
im_ani = animation.ArtistAnimation(fig2, ims, interval=100, repeat_delay=3000, blit=True)
plt.show()
#####
print "norm at the end: "
print sum(abs(psi)**2)*dx
