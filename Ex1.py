import numpy as np
def main():
    w=1
main()


def estimator(file_name):
    avg_monthly_precip = np.genfromtxt(file_name, dtype='str', delimiter=", ")

    nbr_tot = len(avg_monthly_precip)
    nbr_succ = 0
    for a in range(len(avg_monthly_precip)):
        if avg_monthly_precip[a] == ' True':
            nbr_succ = nbr_succ + 1
        if avg_monthly_precip[a] == '[True':
            nbr_succ = nbr_succ + 1
        if avg_monthly_precip[a] == 'True]':
            nbr_succ = nbr_succ + 1
    print(nbr_succ)

    ##MAP estimator
    Z_theta = nbr_succ

    Map_estimator = (Z_theta + 1) / (4 + nbr_tot)
    ML_estimator = Z_theta / nbr_tot
    print("MAP estimator = ", Map_estimator)
    print("ML estimator = ", ML_estimator)

estimator("Estimators1.txt")
estimator("Estimators2.txt")

