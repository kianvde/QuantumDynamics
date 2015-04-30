__author__ = 'kian'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation




fig2, ax = plt.subplots()
ims = []
for i in range(100):
    [psi, garbage] = linalg.bicgstab(A, B*psi)
    im, = plt.plot(x, np.abs(psi)**2, 'b')
    ims.append([im])
im_ani = animation.ArtistAnimation(fig2, ims, interval=100, repeat_delay=3000, blit=True)
plt.show()