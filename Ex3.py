from elevationMap import ElevationMap
from Ex2 import readline_to_vector
import numpy as np
import seaborn as sns
import random  # for random.choices
import matplotlib.pyplot as plt
def main():
    print("Hey")
main()

def sample_multivariate(moyenne, covariance):
    epsilon = 0.0001
    K = covariance + epsilon * np.identity(2)
    L = np.linalg.cholesky(K)
    n = 1
    u = np.random.normal(loc=0, scale=1, size=2*n).reshape(2, n)
    return (moyenne + np.dot(L, u)).reshape(n, 2)


f2 = open("measures2D.txt", 'r')
Yt = readline_to_vector(f2) # Estimation of the ground elevation
f2.readline()
POSITION_X1_t = readline_to_vector(f2) #True position of X1
f2.readline()
POSITION_X2_t = readline_to_vector(f2) #True position of X2
f2.close()

Map = ElevationMap("Ardennes.txt")


t_f = len(Yt)-1  # final time
d_x = 2  # dimension of state space; must be 1 in this script
d_w = 2  # dimension of u; must be 1 in this script
d_y = 1  # dimension of output space; must be 1 in this script

def exercice3():


    mu_w = np.array([0, 0]).reshape(2,1)
    Sigma_w = np.array([[10*10**(-5), 8*10**(-5)], [8*10**(-5), 10*10**(-5)]])
    mu_e = 0
    Sigma_e = np.array([[16]])
    v_t = np.array([-0.4, -0.3])
    delta_t = np.array([0.01, 0.01])

    sqrt_Sigma_e = np.sqrt(Sigma_e)

    out_noise_pdf = lambda w: 1/np.sqrt((2*np.pi)**d_y*np.abs(np.linalg.det(Sigma_e)))* np.exp(-.5*(w-mu_e)@np.linalg.inv(Sigma_e)@(w-mu_e))  # pdf of the output noise w_t
    # *** SEQUENTIAL MONTE CARLO METHOD ***

    n = int(200)  # sample set size. Sugg: 1e2
    X = np.zeros((d_x, n, t_f + 1))  # particles will be stored in X
    Xtilde = np.zeros((d_x, n, t_f + 1))  # to store the predictions

    # ** Generate initial sample set {x_0^i,...,x_0^n}:

    t = 0
    for i in range(n):
        X[0, i, t] = np.array(np.random.uniform(0.2,0.9))  # we sample from the distribution of x_0  # HIDDEN
        X[1, i, t] = np.array(np.random.uniform(0,1))

    # ** Start loop on time:

    for t in range(t_f):

        #print(t)
        # ** Prediction

        for i in range(n):
            w = sample_multivariate(mu_w, Sigma_w) # HIDDEN
            print('w =', w)
            Xtilde[:, i, t + 1] = X[:, i, t] + delta_t*v_t + w  # HIDDEN

        # ** Update


        weights = np.zeros(n)
        for i in range(n):
            weights[i] = out_noise_pdf(np.array([Yt[t+1] - Map.h(Xtilde[:, i, t + 1])] ) ) # HIDDEN

        # Resample the particles according to the weights:
        ind_sample = random.choices(population=np.arange(n), weights=weights, k=n)

        for i in range(n):
            X[:, i, t + 1] = Xtilde[:, ind_sample[i], t + 1]

    # end for t


    # ** Visualization
    plt.figure(1)
    test = np.zeros((1,t_f+1))
    for t in range(t_f + 1):
        # HIDDEN[[
        # Display particles at each time:

        #for i in range(n):
            #plt.plot(t, X[0, i, t], 'ro', markersize=1) #red o


        # Display true x at each time:
        #plt.plot(t, POSITION_X1_t[t], 'kx') #x black
        # Display true y at each time:
        #plt.plot(t, Yt[t], 'k>')

        # Compute and display sample mean for each time:
        x_mean = 0
        for i in range(n):
            x_mean = x_mean + X[1, i, t]

        x_mean = x_mean / n
        test[0,t] = x_mean
        #plt.plot(t, x_mean[0], 'rx')
        # HIDDEN]]

    #plt.xlabel('t')
    #plt.ylabel('x_t^i, i=1,...,n')
    #plt.title('Sequential Monte Carlo experiment')
    #plt.show()
    return test

nbr_trial = 10
mean_vect = np.zeros((1,t_f+1))
for a in range(nbr_trial):
    print(a)
    mean_vect = mean_vect+exercice3()
mean_vect = mean_vect/nbr_trial
for t in range(t_f + 1):
    print(t)
    # HIDDEN[[
    # Display particles at each time:

    # Display true x at each time:
    plt.plot(t, POSITION_X2_t[t], 'kx')  # x black
    # Display true y at each time:
    # plt.plot(t, Yt[t], 'k>')

    # Compute and display sample mean for each time:
    plt.plot(t, mean_vect[0,t], 'rx')
    # plt.plot(t, x_mean[0], 'rx')
    # HIDDEN]]

plt.xlabel('t')
plt.ylabel('x_t^i, i=1,...,n')
plt.title('Sequential Monte Carlo experiment')
plt.show()