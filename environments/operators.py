import numpy as np

I = np.identity(2, dtype=np.complex64)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
S = np.array([[1, 0], [0, 1j]])
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
T = np.array([[1, 0], [0, np.exp((1j * np.pi) / 4)]])
CNOT = np.array([[[[1, 0],[0, 1]], [[0 ,0],[0 ,0]]], [[[0 ,0],[0, 0]],[[0, 1],[1, 0]]]])
SWAP = np.array([[[[1, 0],[0, 0]], [[0 ,0],[1 ,0]]], [[[0 ,1],[0, 0]],[[0, 0],[0, 1]]]])

