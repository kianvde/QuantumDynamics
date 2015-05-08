import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import sys
from time import gmtime, strftime


# Crank-Nicolson method QD time evolution simulation of a gaussian
# wave packet incident on a rectangular potential barrier
def main():

    ## simulation constants ##

    # epsilon is E/v0
    epsilon = np.linspace(0.8, 3.0, 301)

    v0 = 6.125  # barrier height,
    a = 2.      # barrier width
    sigma = 22. # gaussian wave packet spread
    dt = .1     # time step
    dx = .1     # spatial grid distance
    L = 300.    # simulation space length

    animate_and_print = False   # animate the time evolution
    plotTransmission = True     # plot the transmission
    saveTransmission = True     # save the transmission
    shape = "blok"              # pick from 'triangle' and 'blok'

    T = np.zeros(len(epsilon))
    x = np.linspace(0, L, np.floor(L/dx))
    v = potential(x, v0, a, L, shape)

    # loop over epsilon
    for i,k in enumerate(np.sqrt(2*epsilon*v0)):

        print "---- epsilon={eps} ({i}/{n}) ----".format(eps=round(epsilon[i],3),i=i+1,n=len(epsilon))
        psi = init_psi(x, (L-a)/2 - 3*sigma, sigma, k, dx)
        A, B = calculate_AB(dx, dt, v)

        if animate_and_print:
            fig, ax ,ims = init_animation(psi, x, v, v0)
            psi, T[i] = run(A, B, x, psi, a, L, dx, ims)
            print_norms(psi, a, L, dx)
            im_ani = animation.ArtistAnimation(fig, ims, interval=10, repeat_delay=500, blit=True)
            plt.show()
        else:
            psi, T[i] = run(A, B, x, psi, a, L, dx)

    if plotTransmission: plot_transmission(epsilon, T, shape, saveTransmission)


###############
## functions ##
###############

# calculate the matrices for the Crank-Nicolson method
# A = Identity + constant*Hamiltonian
# B = Identity - constant*Hamiltonian
def calculate_AB(dx, dt, v):
    n = len(v)

    # two constants to simplify the expressions
    c1 = 1.j * dt / (4. * dx**2)
    c2 = 1.j * dt / 2.

    A = sparse.diags(1.+2.*c1+c2*v, 0) - c1*sparse.eye(n,n,1) - c1*sparse.eye(n,n,-1)
    B = sparse.diags(1.-2.*c1-c2*v, 0) + c1*sparse.eye(n,n,1) + c1*sparse.eye(n,n,-1)

    # fill in for periodic boundary conditions
    A = A + sparse.diags([[-c1, 0],[0, -c1]],[n-2, 2-n])
    B = B + sparse.diags([[c1, 0],[0, c1]],[n-2, 2-n])

    return A, B

# run the time evolution and return the transmission by solving the
# matrix equation A*psi(t+dt) = B*psi(t) using the bicgstab method.
# The function stops time stepping and calculates the transmission
# when the norm in the barrier has reduced below 10e-6
def run(A, B, x, psi, a, L, dx, ims = None):

    y = 2*np.max(abs(psi))**2
    c = cn = i = 0
    while cn>=c or c>10e-6:
        c = cn
        if ims!=None and i%4==0:
            plt.axis((0,L,0,y))
            im, = plt.plot(x, np.abs(psi)**2, 'b')
            ims.append([im])

        # use the sparse stabilized biconjugate gradient method to solve the matrix eq
        [psi, error] = linalg.bicgstab(A, B*psi, x0=psi)
        if error != 0: sys.exit("bicgstab did not converge")

        i = i+1
        # calculate the new norm in the barrier
        cn = sum(abs(psi[int(round((L - a)/(2*dx))):int(round((L + a)/(2*dx)))])**2)*dx

    return psi, sum(abs(psi[int(round((L - a)/(2*dx))):])**2)*dx/(sum(abs(psi)**2)*dx)

# initialize the wave function
def init_psi(x, x0, sigma, k, dx):
    # create a gaussian wave function moving in the positive x direction
    psi = np.exp(-(1/(2*sigma**2))*(x-x0)**2) * np.exp(1.j*k*x)
    # normalize the wave function
    psi /= np.sqrt(sum(abs(psi)**2*dx))

    return psi

# initialize a rectangular potential barrier at position s
def potential(x, v0, a, L, shape):
    if shape.lower() == 'triangle':
        v = (v0/a)*(x+(a-L)/2)*(abs(x-L/2) < a/2)
    else:
        v = v0*(abs(x-L/2) < a/2)

    return v

# initialize the animation
def init_animation(psi, x, v, v0):
    fig, ax = plt.subplots()
    plt.plot(x, 1.5*max(abs(psi)**2)*v/v0, 'r')
    ims = []

    return fig, ax, ims

# prints norms after every time evolution
def print_norms(psi, a, L, dx):
    print "norm wave function : ",
    print sum(abs(psi)**2)*dx
    print "norm left of barrier: ",
    print sum(abs(psi[0:int(round((L - a)/(2*dx)))])**2)*dx
    print "norm right of barrier: ",
    print sum(abs(psi[int(round((L - a)/(2*dx))):])**2)*dx
    print "approximation Transmission: ",
    print sum(abs(psi[int(round((L - a)/(2*dx))):])**2)*dx/(sum(abs(psi)**2)*dx)

# plots the transmission after the run
def plot_transmission(epsilon, T, shape, saveTransmission):
    plt.figure()
    plt.title("Transmission of a gaussian wave packet \n through a {s} potential barrier".format(s=shape))
    plt.xlabel('epsilon = E/$V_0$')
    plt.ylabel('Transmission')
    plt.axis((0, np.max(epsilon), 0, 1.1))
    plt.axhline(y=1, linewidth=2, color='r')
    plt.vlines(1, 0, 1, color='g', linestyle='--')
    plt.plot(epsilon, T)
    if saveTransmission:
        plt.savefig("{s}.pdf".format(s=strftime("%d-%m-%Y_%H-%M", gmtime())))
    plt.show()

##############
##   main   ##
##############
main()