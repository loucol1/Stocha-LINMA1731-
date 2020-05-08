from elevationMap import ElevationMap
import numpy as np
import random  # for random.choices
import matplotlib.pyplot as plt
def main():
    print("coucou")
main()

"""
Read a line of the file and return the float vector corresponding to this line 
At the end of the function, the read head is on the next line
"""
def readline_to_vector(file):
    r = file.read(1)
    while(r != '['): #position at the begining of the vector
        r = file.read(1)
    retour = file.readline()
    retour = retour[0:len(retour)-2] #delete the ']' at the end of the vector
    retour = retour.split(", ")
    retour = list(map(float, retour)) # return the float vector
    return retour




f1 = open("measures1D.txt", 'r')
Y_t = readline_to_vector(f1) # Estimation of the ground elevation
f1.readline()
POSITION_t = readline_to_vector(f1) #True position of the plane
f1.close()

Map = ElevationMap("Ardennes.txt")



t_f = len(Y_t)-1  # final time
d_x = 1  # dimension of state space;
d_w = 1  # dimension of w;


def exercice2(n):



    mu_w = 0
    Sigma_w = np.array([[0.004]])
    mu_e = 0
    Sigma_e = np.array([[16]])
    v_t = 1.6
    delta_t = 0.01 #because 100 measurements per second so delta_t =1/100 second between the measurements


    sqrt_Sigma_w = np.sqrt(Sigma_w)
    sqrt_Sigma_e = np.sqrt(Sigma_e)

    out_noise_pdf = lambda w: 1 /(sqrt_Sigma_e * np.sqrt((2 * np.pi)) )* np.exp(
        -.5 * ((w - mu_e)/sqrt_Sigma_e)**2)  # pdf of the output noise e_t



    # *** SEQUENTIAL MONTE CARLO METHOD ***

    #n = int(1e2)  # sample set size.
    X = np.zeros((d_x, n, t_f + 1))  # particles will be stored in X
    Xtilde = np.zeros((d_x, n, t_f + 1))  # to store the predictions

    # ** Generate initial sample set {x_0^i,...,x_0^n}:

    t = 0
    for i in range(n):
        X[:, i, t] = np.random.uniform(0,1)  # we sample from the distribution of x_0

    # ** Start loop on time:

    for t in range(t_f):

        for i in range(n):
            w = mu_w + sqrt_Sigma_w * np.random.randn(d_w, 1)
            Xtilde[:, i, t + 1] = X[:, i, t] + delta_t*v_t + w

        # ** Update


        weights = np.zeros(n)
        for i in range(n):
            weights[i] = out_noise_pdf(Y_t[t+1] - Map.h(float(Xtilde[0, i, t + 1])) )  # The weights are chosen from the pdf of the noise e_t

        # Resample the particles according to the weights:
        ind_sample = random.choices(population=np.arange(n), weights=weights, k=n)

        for i in range(n):
            X[:, i, t + 1] = Xtilde[:, ind_sample[i], t + 1]
    # end for t


    # ** Visualization
    #plt.figure(1)
    x_mean_vector = np.zeros(t_f+1)
    for t in range(t_f + 1):

        #for i in range(n):
        #    plt.plot(t, X[0, i, t], 'ro', markersize=1) #red o for all the particles

        # Display true x at each time:
        #plt.plot(t, POSITION_t[t], 'kx') #x black for the true position


        # Compute and display sample mean for each time:
        x_mean = 0
        for i in range(n):
            x_mean = x_mean + X[0, i, t]

        x_mean = x_mean / n
        x_mean_vector[t] = x_mean
        #plt.plot(t, x_mean[0], 'rx')
    return x_mean_vector
    #plt.xlabel('t')
    #plt.ylabel('x_t^i, i=1,...,n')
    #plt.title('Sequential Monte Carlo experiment')
    #plt.show()
nbr_trial = 10
nbr_sample = 10
nbr_vector_sample = np.linspace(10, 100, nbr_sample)
vect_stock = np.zeros((t_f + 1, nbr_sample))
vect_position = POSITION_t * np.ones((nbr_sample, t_f + 1))
vect_position = vect_position.T

for q in range(nbr_trial):
    count = 0
    for u in nbr_vector_sample:
        vect_stock[:, count] = vect_stock[:, count]+ exercice2(int(u))
        count = count + 1

vect_stock = vect_stock/nbr_trial
matrix_error = np.abs(vect_position - vect_stock)
vect_error = np.sum(matrix_error, axis=0)



plt.plot(nbr_vector_sample, vect_error)
plt.show()
