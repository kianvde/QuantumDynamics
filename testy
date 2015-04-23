import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
import numpy as np

# splinalg.bicgstab

s = 1.0/4.0j
dt = 0.1
a = 0.1
L = 1.0
N = L/a
k = 1.0
V = 0

MatImp = (1.0-2.0*s)*sparse.identity(N) + s*sparse.eye(N,N,1) + s*sparse.eye(N,N,-1)
MatExp = (1.0+2.0*s)*sparse.identity(N) - s*sparse.eye(N,N,1) - s*sparse.eye(N,N,-1)

psistart = [0,0,0,0,0,1,0,0,0,0]

psiexp = MatExp * psistart
[psinext , zooi] = splinalg.bicgstab(MatImp , psiexp , psiexp)

probdist = abs(psinext)**2
print probdist

plt.figure(1)
plt.plot(probdist)
plt.show()







