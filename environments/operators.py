import numpy as np

I = np.identity(2, dtype=np.complex64)
II = np.tensordot(I, I, axes=0)
III = np.tensordot(I, II, axes=0)
X = np.array([[0, 1], [1, 0]]).astype(np.complex64)
Y = np.array([[0, -1j], [1j, 0]]).astype(np.complex64)
Z = np.array([[1, 0], [0, -1]]).astype(np.complex64)
S = np.array([[1, 0], [0, 1j]]).astype(np.complex64)
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]]).astype(np.complex64)
T = np.array([[1, 0], [0, np.exp((1j * np.pi) / 4)]]).astype(np.complex64)
Tdag = np.matrix(T).getH() #complex conjugate transpose
CNOT = np.array([[[[1, 0],[0, 1]], [[0 ,0],[0 ,0]]], [[[0 ,0],[0, 0]],[[0, 1],[1, 0]]]]).astype(np.complex64)
P0 = np.array([[1,0],[0,0]]).astype(np.complex64)
P1 = np.array([[0,0],[0,1]]).astype(np.complex64)
TOFFOLI = np.tensordot(P0, II, axes=0) + np.tensordot(P1, CNOT, axes=0)
SWAP = np.array([[[[1, 0],[0, 0]], [[0 ,0],[1 ,0]]], [[[0 ,1],[0, 0]],[[0, 0],[0, 1]]]]).astype(np.complex64)
FREDKIN = np.tensordot(P0, II, axes=0) + np.tensordot(P1, SWAP, axes=0)

