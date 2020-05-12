from elevationMap import ElevationMap
from Ex2 import readline_to_vector
import numpy as np
import random  # for random.choices
import matplotlib.pyplot as plt
import math
def main():
    print("Hey")
main()

f2 = open("measures2D.txt", 'r')
Yt = readline_to_vector(f2)
f2.readline()
POSITION_X1_t = readline_to_vector(f2)
f2.readline()
POSITION_X2_t = readline_to_vector(f2)
f2.close()
Map = ElevationMap("Ardennes.txt")

def cholesky_decomposition(matrix):
    a = math.sqrt(matrix[0,0])
    b = matrix[0,1]/a
    c = math.sqrt(matrix[1,1]-b**(2))
    return np.array([[a,0],[b,c]])

def sample_multivariate(moyenne, covariance):
    epsilon = 0.0001
    K = covariance + epsilon * np.identity(2)
    L = cholesky_decomposition(K)#np.linalg.cholesky(K)
    n = 1
    u = np.random.normal(loc=0, scale=1, size=2*n).reshape(2, n)
    return (moyenne + np.dot(L, u)).reshape(n, 2)
'''
moyenne = np.array([1 ,2]).reshape(2,1)
covariance = np.array([[2,1],[1,2]])
x = sample_multivariate(moyenne, covariance)
print(covariance)
print(x)
sns.jointplot(x=x[0],
              y=x[1],
              kind="kde",
              space=0)
plt.show()
'''


t_f = len(Yt)-1  # final time
d_x = 2  # dimension of state space
d_w = 2  # dimension of u
d_y = 1  # dimension of output space

mu_w = np.array([0, 0]).reshape(2,1)
Sigma_w = np.array([[0.0001, 0.00008], [0.00008, 0.0001]])
mu_e = 0
Sigma_e = np.array([[16]])
v_t = np.array([-0.4, -0.3])
delta_t = np.array([0.01, 0.01])

sqrt_Sigma_e = np.sqrt(Sigma_e)

out_noise_pdf = lambda w: 1 / (sqrt_Sigma_e * np.sqrt((2 * np.pi)**2)) * np.exp(
    -.5 * ((w - mu_e) / sqrt_Sigma_e) ** 2)  # pdf of the output noise w_t
# *** SEQUENTIAL MONTE CARLO METHOD ***

n = int(1e2)  # sample set size. Sugg: 1e2
X = np.zeros((d_x, n, t_f + 1))  # particles will be stored in X
Xtilde = np.zeros((d_x, n, t_f + 1))  # to store the predictions

# ** Generate initial sample set {x_0^i,...,x_0^n}:

t = 0
for i in range(n):
    X[:, i, t] = np.array(np.random.uniform(0,1, size = (1,2))) # we sample from the distribution of x_0
# ** Start loop on time:

for t in range(t_f):

    # ** Prediction

    for i in range(n):
        w = sample_multivariate(mu_w, Sigma_w) # HIDDEN
        #w = w.flatten()
        Xtilde[:, i, t + 1] = X[:, i, t] + delta_t*v_t + w

    # ** Update

    weights = np.zeros(n)
    for i in range(n):
        weights[i] = out_noise_pdf(Yt[t+1] - Map.h(Xtilde[:, i, t + 1]))

    # Resample the particles according to the weights:
    ind_sample = random.choices(population=np.arange(n), weights=weights, k=n)

    for i in range(n):
        X[:, i, t + 1] = Xtilde[:, ind_sample[i], t + 1]

# end for t


# ** Visualization
plt.figure(1)
for t in range(t_f + 1):
    # Display particles at each time:

    for i in range(n):
        plt.plot(t, X[1, i, t], 'ro', markersize=1) #red o
    # Display true x at each time:
    plt.plot(t, POSITION_X2_t[t], 'kx') #x black
    # Display true y at each time:
    #plt.plot(t, Yt[t], 'k>')

plt.xlabel('t')
plt.ylabel('x_t^i, i=1,...,n')
plt.title('Sequential Monte Carlo experiment')
plt.show()


