# Ahmet Ali Nuhoğlu
# 21602149
# EEE482 - Computational Neuroscience Homework 1

import numpy as np  # for easier computations
import matplotlib.pyplot as plt  # for plotting
import math  # for easier computations
import sys

# for displaying all of the outputs in case of a need for debugging
np.set_printoptions(threshold=sys.maxsize)

question = sys.argv[1]

def ahmet_ali_nuhoglu_21602149_hw1(question):
    if question == '1' :

        print("Question 1")
        print("Part A")

        # defining the array A provided in the homework assignment
        A = np.array([[1, 0, -1, 2], [2, 1, -1, 5], [3, 3, 0, 9]])

        # generating two random variables for the free variable values
        x3 = np.random.rand()
        x4 = np.random.rand()

        # interpretation of the solution derived by hand
        x_general = np.array([[x3 - 2 * x4], [-x3 - x4], [x3], [x4]])

        print(np.around(A.dot(x_general).T, 4))

        print("Part B")

        # defining the given matrix A
        A = np.array([[1, 0, -1, 2], [2, 1, -1, 5], [3, 3, 0, 9]])

        # defining a matrix for a particular solution which is derived by hand
        x_particular = np.array([[1], [2], [0], [0]])

        # printing the output on the console. The output has to be equal to the given b matrix
        print(A.dot(x_particular).T)

        print("Part C")

        # defining the given matrix A
        A = np.array([[1, 0, -1, 2], [2, 1, -1, 5], [3, 3, 0, 9]])

        # assigning random values to free variables
        x3 = np.random.rand()
        x4 = np.random.rand()

        # assigning numerical values to the general solution, using the random numbers generated above
        x_general = np.array([[1 + x3 - 2 * x4], [2 - x3 - x4], [x3], [x4]])

        # printing the output on the console. The output has to be equal to the given b matrix
        print(A.dot(x_general).T)

        print("Part D")

        # using the svd function defined in the numpy library to find
        # the elements of the singular value decomposition
        u, s, vt = np.linalg.svd(A)
        print("Left Singular Vector matrix U:\n", u, "\n")
        print("Right Singular Vector matrix V:\n", vt)
        print('\nSigma:')

        # for diagonalizing the sigma values we create a matrix with the
        # same shape with the provided matrix A. Therefore, using a for
        # loop we diagonalize the singular values.
        sigma = np.zeros(A.shape)
        rows, cols = sigma.shape
        for i in range(rows):
            sigma[i][i] = np.around(s[i], 7)
        print(sigma, '\n')

        # finding pseudo inverse of the sigma from the sigma
        sigma_pseudo = np.zeros(sigma.shape)
        for i in range(rows):
            if sigma[i][i] != 0:
                sigma_pseudo[i][i] = 1 / sigma[i][i]
            else:
                sigma_pseudo[i][i] = 0

        sigma_pseudo = sigma_pseudo.T

        A_pseudo = vt.T.dot(sigma_pseudo).dot(u.T)
        print("A = U.Sigma.V_t\n", np.around(u.dot(sigma).dot(vt)))
        print('\n A pseudo inverse:\n', A_pseudo)
        print('\n A(A^T)A: \n', np.around(A.dot(A_pseudo).dot(A)))

        # validating the previous result from the built in
        # function to find pseudo inverse of A
        A_pinv = np.linalg.pinv(A)
        print("A pseudo inverse: \n", A_pinv, "\n")
        print("A*A_pinv*A: \n", np.around(A.dot(A_pinv).dot(A)))

        print("Part E")

        # found sparsest solutions
        xs_1 = [[1], [2], [0], [0]]
        xs_2 = [[0], [0], [1], [1]]
        xs_3 = [[0], [1.5], [0], [0.5]]
        xs_4 = [[-3], [0], [0], [2]]
        xs_5 = [[0], [3], [-1], [0]]
        xs_6 = [[3], [0], [2], [0]]

        print("A.xs_1 = ", A.dot(xs_1).T)
        print("A.xs_2 = ", A.dot(xs_2).T)
        print("A.xs_3 = ", A.dot(xs_3).T)
        print("A.xs_4 = ", A.dot(xs_4).T)
        print("A.xs_5 = ", A.dot(xs_5).T)
        print("A.xs_6 = ", A.dot(xs_6).T)

        print("Part F")

        # Here we find the least norm solution using
        # pseudo inverse of matrix A
        B = np.array([[1], [4], [9]])
        x_least_norm = A_pinv.dot(B)
        print("Least norm solution:", x_least_norm.T)

    elif question == '2':

        print("Question 2")

        print("Part A")

        # using arrange function from the numpy library to generate a data set with
        # the values wanted form us in the assignment
        data_points = np.arange(0, 1.001, 0.001)
        x_l = bernuolli(data_points, 869, 103)
        x_nl = bernuolli(data_points, 2353, 199)

        plt.figure()
        plt.bar(np.arange(len(data_points)), x_l, color='g')
        plt.xlim(0, 200)
        plt.xticks(np.arange(0, 201, 20), np.around(np.arange(0, 0.201, 0.02), 2))
        plt.title("Likelihood function of tasks involving language")
        plt.xlabel("Probability")
        plt.ylabel("Likelihood")

        plt.figure()
        plt.xlim(0, 200)
        plt.xticks(np.arange(0, 201, 20), np.around(np.arange(0, 0.201, 0.02), 2))
        plt.title("Likelihood function of tasks not involving language")
        plt.xlabel("Probability")
        plt.ylabel("Likelihood")
        plt.bar(np.arange(len(data_points)), x_nl, color='r')
        plt.show(block=False)

        print("Part B")

        print("Maximum likelihood of x_l and its probability respectively: (", np.amax(x_l), ", ",
              np.argmax(x_l) / 1000, ")")
        print("Maximum likelihood of x_nl and its probability respectively: (", np.amax(x_nl), ", ",
              np.argmax(x_nl) / 1000, ")")

        print("Part C")

        # (given in the assignment) uniform prior
        uniform_pri = 1 / len(data_points)

        # computing the normalizer with the previors values found
        xl_normalizer = np.sum(x_l * uniform_pri)
        xnl_normalizer = np.sum(x_nl * uniform_pri)

        # computing the posterior distributions with the obtained
        # values
        xl_posterior = x_l * uniform_pri / xl_normalizer
        xnl_posterior = x_nl * uniform_pri / xnl_normalizer

        # creating numpy arrays to store the cdf's of the posteriors
        xl_posterior_cdf = np.zeros(len(data_points))
        xnl_posterior_cdf = np.zeros(len(data_points))

        # booleans to find the bounds of the 95% CI of tasks
        # involving language, using CDF
        xl_lower = -1
        found_xl_lower = False
        xl_upper = -1
        found_xl_upper = False

        # computing the CDF of the posterior function of tasks
        # involving language and holding the CI bounds
        for i in range(0, len(data_points)):
            if i == 0:
                xl_posterior_cdf[i] = xl_posterior[i]
            else:
                xl_posterior_cdf[i] = xl_posterior_cdf[i - 1] + xl_posterior[i]

            if xl_posterior_cdf[i] > 0.025 and not found_xl_lower:
                xl_lower = np.round(data_points[i], 3)
                found_xl_lower = True
            elif xl_posterior_cdf[i] > 0.975 and not found_xl_upper:
                xl_upper = np.round(data_points[i], 3)
                found_xl_upper = True

        # booleans to find the bounds of the 95% CI of tasks
        # involving language, using CDF
        xnl_lower = -1
        found_xnl_lower = False
        xnl_upper = -1
        found_xnl_upper = False

        # computing the CDF of the posterior function of tasks
        # not involving language and holding the CI bounds
        for i in range(0, len(data_points)):
            if i == 0:
                xnl_posterior_cdf[i] = xnl_posterior[i]
            else:
                xnl_posterior_cdf[i] = xnl_posterior_cdf[i - 1] + xnl_posterior[i]

            if xnl_posterior_cdf[i] > 0.025 and not found_xnl_lower:
                xnl_lower = np.round(data_points[i], 3)
                found_xnl_lower = True
            elif xnl_posterior_cdf[i] > 0.975 and not found_xnl_upper:
                xnl_upper = np.round(data_points[i], 3)
                found_xnl_upper = True

        # Plotting the figures below
        plt.figure()
        plt.xlim(0, 200)
        plt.xticks(np.arange(0, 201, 20), np.around(np.arange(0, 0.201, 0.02), 2))
        plt.title("Posterior distribution of tasks not involving language")
        plt.xlabel("Probability")
        plt.ylabel("P(X|data) (Posterior)")
        plt.bar(np.arange(len(data_points)), xl_posterior, color='g')

        plt.figure()
        plt.xlim(0, 200)
        plt.xticks(np.arange(0, 201, 20), np.around(np.arange(0, 0.201, 0.02), 2))
        plt.title("Posterior distribution of tasks not involving language")
        plt.xlabel("Probability")
        plt.ylabel("P(X|data) (Posterior)")
        plt.bar(np.arange(len(data_points)), xnl_posterior, color='r')

        plt.figure()
        plt.xticks(np.arange(0, 1001, 100), np.around(np.arange(0, 1.001, 0.1), 2))
        plt.title("Cumulative distribution function of tasks involving language")
        plt.xlabel("Probability")
        plt.ylabel("CDF")
        plt.bar(np.arange(len(data_points)), xl_posterior_cdf, color='g')

        plt.figure()
        plt.xticks(np.arange(0, 1001, 100), np.around(np.arange(0, 1.001, 0.1), 2))
        plt.title("Cumulative distribution function of tasks not involving language")
        plt.xlabel("Probability")
        plt.ylabel("CDF")
        plt.bar(np.arange(len(data_points)), xnl_posterior_cdf, color='r')
        plt.show(block=False)

        print("%95 Confidence Interval Bounds for tasks involving language")
        print("\tLower: ", xl_lower, "\tHigher: ", xl_upper, "\n")

        print("%95 Confidence Interval Bounds for tasks not involving language")
        print("\tLower: ", xnl_lower, "\tHigher: ", xnl_upper)

        print("Part D")

        # we put our posterior distributions in matrix form for matrix
        # multiplication in order to find the joint distributions
        xl_matrix = np.matrix(xl_posterior)
        xnl_matrix = np.matrix(xnl_posterior).transpose()

        joint_matrix = np.matrix(xnl_matrix * xl_matrix)

        # plotting the joint posterior distribution
        plt.figure()
        plt.imshow(joint_matrix, origin='lower')
        plt.colorbar()
        plt.xlabel("$x_l$")
        plt.ylabel("$x_{nl}$")
        plt.title("Joint Posterior Distribution\n $P(X_l, X_{nl}|data)$")
        plt.show(block=False)

        xl_greater = 0
        xnl_greater = 0

        # computing the P(X_l > X_nl | data) and P(X_nl >= X_l | data)
        for i in range(len(xl_posterior)):
            for j in range(len(xnl_posterior)):
                if i > j:
                    xnl_greater += joint_matrix[i, j]
                else:
                    xl_greater += joint_matrix[i, j]

        print("P(X_l > X_nl | data) = ", np.around(xl_greater, 4))
        print("P(X_nl >= X_l | data) = ", np.around(xnl_greater, 4))

        print("Part E")

        xl_prob = xnl_prob = .5

        # using the Bayes rule we find the P(language|activation) by reverse inference
        activation_prob = np.argmax(x_l) / 1000 * xl_prob + np.argmax(x_nl) / 1000 * xnl_prob
        reverse_inference = (np.argmax(x_l) / 1000 * xl_prob) / activation_prob
        print("P(language|activation) = ", np.around(reverse_inference, 4))

# this function is used to generate a dataset with bernuolli distribution, with given parameters
def bernuolli(x, total_trials, positive_results):
    comb = (math.factorial(total_trials) / (
            math.factorial(positive_results) * math.factorial(total_trials - positive_results)))
    return comb * (x ** positive_results) * ((1 - x) ** (total_trials - positive_results))

ahmet_ali_nuhoglu_21602149_hw1(question)
