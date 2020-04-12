from elevationMap import ElevationMap
from Ex2 import readline_to_vector
import numpy as np
import seaborn as sns
import random  # for random.choices
import matplotlib.pyplot as plt
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

def sample_multivariate(moyenne, covariance):
    epsilon = 0.0001
    K = covariance + epsilon * np.identity(2)
    L = np.linalg.cholesky(K)
    n = 10000
    u = np.random.normal(loc=0, scale=1, size=2*n).reshape(2, n)
    return moyenne + np.dot(L, u)

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
d_x = 1  # dimension of state space; must be 1 in this script
d_y = 1  # dimension of output space; must be 1 in this script
d_w = 1  # dimension of u; must be 1 in this script


mu_x = 0  # see definition above
Sigma_x = np.array([[1]])  # The current version requires a 2d-array but it assumes a (1, 1) shape
mu_w = 0
Sigma_w = np.array([[0.004]])
mu_e = 0
Sigma_e = np.array([[16]])
v_t = [-0.4, -0.3]
delta_t = 0.01

sqrt_Sigma_x = np.sqrt(Sigma_x)
sqrt_Sigma_w = np.sqrt(Sigma_w)
sqrt_Sigma_e = np.sqrt(Sigma_e)

out_noise_pdf = lambda w: 1 /(sqrt_Sigma_e * np.sqrt((2 * np.pi)) )* np.exp(
    -.5 * ((w - mu_e)/sqrt_Sigma_e)**2)  # pdf of the output noise w_t



# *** SEQUENTIAL MONTE CARLO METHOD ***

n = int(1e2)  # sample set size. Sugg: 1e2
X = np.zeros((d_x, n, t_f + 1))  # particles will be stored in X
Xtilde = np.zeros((d_x, n, t_f + 1))  # to store the predictions

# ** Generate initial sample set {x_0^i,...,x_0^n}:

t = 0
for i in range(n):
    X[:, i, t] = mu_x + sqrt_Sigma_x * np.random.randn(d_x, 1)  # we sample from the distribution of x_0  # HIDDEN

# ** Start loop on time:

for t in range(t_f):

    #print(t)
    # ** Prediction

    for i in range(n):
        w = mu_w + sqrt_Sigma_w * np.random.randn(d_w, 1)  # HIDDEN
        Xtilde[:, i, t + 1] = X[:, i, t] + delta_t*v_t + w  # HIDDEN

    # ** Update


    weights = np.zeros(n)
    for i in range(n):
        weights[i] = out_noise_pdf(Y_t[t+1] - Map.h(float(Xtilde[0, i, t + 1])) )  # HIDDEN

    # Resample the particles according to the weights:
    ind_sample = random.choices(population=np.arange(n), weights=weights, k=n)

    for i in range(n):
        X[:, i, t + 1] = Xtilde[:, ind_sample[i], t + 1]

# end for t


# ** Visualization
plt.figure(1)
for t in range(t_f + 1):
    print(t)
    # HIDDEN[[
    # Display particles at each time:
    '''
    #for i in range(n):
    #    plt.plot(t, X[0, i, t], 'ro', markersize=1) #red o
'''
    # Display true x at each time:
    plt.plot(t, POSITION_t[t], 'kx') #x black
    # Display true y at each time:

    plt.plot(t, Y_t[t], 'k>')

    # Compute and display sample mean for each time:
    x_mean = np.zeros((d_x, 1))
    for i in range(n):
        x_mean = x_mean + X[0, i, t]

    x_mean = x_mean / n
    plt.plot(t, x_mean[0], 'rx')
    # HIDDEN]]

plt.xlabel('t')
plt.ylabel('x_t^i, i=1,...,n')
plt.title('Sequential Monte Carlo experiment')
plt.show()
'''
