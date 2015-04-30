import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
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
fig2 = plt.figure()
ax = fig2.add_subplot(111,projection='3d')
ims = []
x= np.linspace(0,1,n)
y = x
xplot , yplot = np.meshgrid(x,y)
# x = np.repeat(x,n)
# x = x.reshape(n,n)
# y = x.transpose()
for i in range(100):
    [psi, garbage] = linalg.bicgstab(A, B*psi)
    psi2 = psi.reshape(n,n)
    im = ax.plot_surface(xplot,yplot,np.abs(psi2)**2),
    ims.append(im)
im_ani = animation.ArtistAnimation(fig2, ims, interval=1000, repeat_delay=3000, blit=True)
plt.show()

#####
# [psi, garbage] = linalg.bicgstab(A, B*psi)
# psi = psi.reshape(n,n)
#
# plt.figure()
# plt.pcolor(abs(psi)**2)