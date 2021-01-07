import numpy as np

I = np.identity(2, dtype=np.complex64) #I = np.array([[1,0],[0,1]])
II = np.tensordot(I, I, axes=0)
X = np.array([[0, 1], [1, 0]]).astype(np.complex64) #X = np.array([[0,1],[1,0]])
Y = np.array([[0, -1j], [1j, 0]]).astype(np.complex64)
Z = np.array([[1, 0], [0, -1]]).astype(np.complex64)
S = np.array([[1, 0], [0, 1j]]).astype(np.complex64)
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]]).astype(np.complex64)
T = np.array([[1, 0], [0, np.exp((1j * np.pi) / 4)]]).astype(np.complex64)
Tdag = np.matrix(T).getH() #complex conjugate transpose
CNOT = np.array([[[[1, 0],[0, 1]], [[0 ,0],[0 ,0]]], [[[0 ,0],[0, 0]],[[0, 1],[1, 0]]]]).astype(np.complex64)
#cnot = np.tensordot(P0, I, axes=((),())) + np.tensordot(P1, X, axes=((),()))
P0 = np.array([[1,0],[0,0]]) #what???
P1 = np.array([[0,0],[0,1]])
TOFFOLI = np.tensordot(P0, II, axes=0) + np.tensordot(P1, CNOT, axes=0)


SWAP = np.array([[[[1, 0],[0, 0]], [[0 ,0],[1 ,0]]], [[[0 ,1],[0, 0]],[[0, 0],[0, 1]]]]).astype(np.complex64)

