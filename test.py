import numpy as np
import math

def cholesky_decomposition(matrix):
    a = math.sqrt(matrix[0,0])
    b = matrix[0,1]/a
    c = math.sqrt(matrix[1,1]-b**(2))
    return np.array([[a,0],[b,c]])


test = np.array([[1,6],[2,13]],float)
print('cho1 = ', cholesky_decomposition(test))
print('cho2 = ', np.linalg.cholesky(test))
