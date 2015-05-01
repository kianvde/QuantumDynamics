import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# simulation constants
dt = 0.025
dx = 0.01
L = 60.0
k = 5.
n = np.floor(L/dx)

# two constants following from two simplify the expressions
c1 = 1.j * dt / (4. * dx**2)    # i*hbar*dt/(4*m*dx^2)
c2 = 1.j * dt / 2.              # i*dt/(2*hbar)

# generate x-coordinates and initial wave-function
x = np.linspace(0,L,n)
psi = np.exp(-0.1*(x-L/4)**2) * np.exp(1.j*k*x)
psi /= np.sqrt(sum(abs(psi)**2*dx)) # normalize the wave function

# set a potential
v = 1.0*np.zeros(n)
v[2972:3028] = 12.5
# this barrier of width a = 56*dx = 0.56 should give a transmission of ~1/3
# E = V0 (=(k^2)/2=12.5) yields: T = 1/(1+0.5*V*a^2)

# numerically solve for the matrix equation A*psi(t+dt) = B*psi(t) where A and B are tridiagonal
A = sparse.diags(1.+2.*c1+c2*v, 0) - c1*sparse.eye(n,n,1) - c1*sparse.eye(n,n,-1)
B = sparse.diags(1.-2.*c1-c2*v, 0) + c1*sparse.eye(n,n,1) + c1*sparse.eye(n,n,-1)
# fill in for periodic boundary conditions
A = A + sparse.diags([[-c1, 0],[0, -c1]],[n-2, 2-n])
B = B + sparse.diags([[c1, 0],[0, c1]],[n-2, 2-n])


fig2, ax = plt.subplots()
plt.vlines([L*(0.5-28/3000), L*(0.5+28/3000)], 0, 1)
ims = []
for i in range(200):

    # use the sparse stabilized biconjugate gradient method to solve the matrix eq
    [psi, garbage] = linalg.bicgstab(A, B*psi)

    if i%5 == 0:
        im, = plt.plot(x[2000:4000], np.abs(psi[2000:4000])**2, 'b')
        ims.append([im])

print "norm at the end: ",
print sum(abs(psi)**2)*dx
print "norm left: ",
print sum(abs(psi[0:2999])**2)*dx
print "norm right: ",
print sum(abs(psi[3000:])**2)*dx
print "approximation Transmission: ",
print sum(abs(psi[3000:])**2)*dx/(sum(abs(psi)**2)*dx)

# animate after enter #
try:
    input("Press enter to continue")
except SyntaxError:
    pass
im_ani = animation.ArtistAnimation(fig2, ims, interval=100, repeat_delay=3000, blit=True)
plt.show()

