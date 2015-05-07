import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# simulation constants
dt = 0.05
dx = 0.01
L = 100.0
v0 = 6.125
E = np.linspace(0.8,1.4,10)*v0
a = 2.


k_vector = np.sqrt(2*E)
transmission = np.zeros(len(k_vector))
n = np.floor(L/dx)
for i,k in enumerate(k_vector):

    # two constants to simplify the expressions
    c1 = 1.j * dt / (4. * dx**2)    # i*hbar*dt/(4*m*dx^2)
    c2 = 1.j * dt / 2.              # i*dt/(2*hbar)

    # generate x-coordinates and initial wave-function
    x = np.linspace(0,L,n)
    psi = np.exp(-0.005*(x-.25*L)**2) * np.exp(1.j*k*x)
    psi /= np.sqrt(sum(abs(psi)**2*dx))  # normalize the wave function

    # set a potential
    v = np.zeros(n)
    v[int(round((L - a)/(2*dx))):int(round((L + a)/(2*dx)))] = v0
    # this barrier of width a = 56*dx = 0.56 should give a transmission of ~1/3
    # E = V0 (=(k^2)/2=12.5) yields: T = 1/(1+0.5*V*a^2)

    # numerically solve for the matrix equation A*psi(t+dt) = B*psi(t) where A and B are tridiagonal
    A = sparse.diags(1.+2.*c1+c2*v, 0) - c1*sparse.eye(n,n,1) - c1*sparse.eye(n,n,-1)
    B = sparse.diags(1.-2.*c1-c2*v, 0) + c1*sparse.eye(n,n,1) + c1*sparse.eye(n,n,-1)
    # fill in for periodic boundary conditions
    A = A + sparse.diags([[-c1, 0],[0, -c1]],[n-2, 2-n])
    B = B + sparse.diags([[c1, 0],[0, c1]],[n-2, 2-n])

    print "----calculating:",
    print "E/V0 =",
    print E[i]/v0,
    print "----"

    fig, ax = plt.subplots()
    plt.vlines([(L-a)/2, (L+a)/2], 0, 2*max(abs(psi)**2))
    ims = []

    c = 0
    c_old = 0
    j = 0
    while c_old<=c or c>10e-4:
        c_old = c
        if j%10 == 0:
            im, = plt.plot(x, np.abs(psi)**2, 'b')
            ims.append([im])

        # use the sparse stabilized biconjugate gradient method to solve the matrix eq
        [psi, garbage] = linalg.bicgstab(A, B*psi, x0=psi)

        j += 1
        c = sum(abs(psi[int(round((L - a)/(2*dx))):int(round((L + a)/(2*dx)))])**2)*dx

    print "norm at the end: ",
    print sum(abs(psi)**2)*dx
    print "norm left: ",
    print sum(abs(psi[0:int(round((L - a)/(2*dx)))])**2)*dx
    print "norm right: ",
    print sum(abs(psi[int(round((L - a)/(2*dx))):])**2)*dx
    transmission[i] = sum(abs(psi[int(round((L - a)/(2*dx))):])**2)*dx/(sum(abs(psi)**2)*dx)
    print "approximation Transmission: ",
    print transmission[i]

    im_ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=3000, blit=True)
    plt.show()

plt.figure()
plt.plot(E/v0, transmission)
plt.show()