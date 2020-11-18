# Ahmet Ali Nuhoğlu
# 21602149

# EEE482 - Computational Neuroscience Homework 3

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.stats import norm

question = sys.argv[1]

def ahmet_ali_nuhoglu_21602149_hw2(question):
    if question == '1' :

        print("Question 1")
        print("Part A")

        with h5py.File('hw3_data2.mat', 'r') as file:
            Xn, Yn = list(file['Xn']), list(file['Yn'])

        Xn = np.array(Xn).T
        Yn = np.array(Yn).flatten()

        def ridge_regression(X, y, lmbd):
            return np.linalg.inv(X.T.dot(X) + lmbd * np.identity(np.shape(X)[1])).dot(X.T).dot(y)

        def r_squared(Y, pred):
            return (np.corrcoef(Y, pred)[0, 1]) ** 2

        def cross_validation(X, y, K, lmbd):

            part_len = int(np.size(y) / K)

            valid_means_d = dict()
            test_means_d = dict()

            for i in range(K):
                valid_data_start = i * part_len
                test_data_start = (i + 1) * part_len
                train_data_start = (i + 2) * part_len

                train_data_ind, test_data_ind, valid_data_ind = [], [], []

                for j in range(valid_data_start, test_data_start):
                    valid_data_ind.append(j % np.size(y))

                for j in range(test_data_start, train_data_start):
                    test_data_ind.append(j % np.size(y))

                for j in range(train_data_start, valid_data_start + np.size(y)):
                    train_data_ind.append(j % np.size(y))

                x_valid, x_test, x_train = X[valid_data_ind], X[test_data_ind], X[train_data_ind]
                y_valid, y_test, y_train = y[valid_data_ind], y[test_data_ind], y[train_data_ind]

                for l in lmbd:
                    weight = ridge_regression(x_train, y_train, l)

                    valid_means_d.setdefault(l, []).append(r_squared(y_valid, x_valid.dot(weight)))
                    test_means_d.setdefault(l, []).append(r_squared(y_test, x_test.dot(weight)))

            valid_means_d = dict((lmbd, np.mean(val)) for lmbd, val in valid_means_d.items())
            test_means_d = dict((lmbd, np.mean(val)) for lmbd, val in test_means_d.items())

            return valid_means_d, test_means_d

        lambda_values = np.logspace(0, 12, num=500, base=10)
        dict_valid, dict_test = cross_validation(Xn, Yn, 10, lambda_values)

        lambda_opt = max(dict_valid, key=lambda k: dict_valid[k])

        x_val, y_val = zip(*sorted(dict_valid.items()))
        x_tst, y_tst = zip(*sorted(dict_test.items()))

        plt.figure()
        plt.plot(x_tst, y_tst)
        plt.plot(x_val, y_val)
        plt.legend(['Test Data', 'Validation Data', ])
        plt.ylabel(r'$R^2$')
        plt.xlabel(r'$\lambda$')
        plt.title(r'$R^2$'' vs ''$\lambda$')
        plt.xscale('log')
        plt.grid()
        plt.show(block=False)

        print("Optimal Lambda Value: ", lambda_opt)

        print("Part B")

        np.random.seed(3)

        def bootstrap(iter_num, x, y, lmbd):
            weight_new = []
            for i in range(iter_num):
                new_ind = np.random.choice(np.arange(np.size(y)), np.size(y))
                x_new, y_new = Xn[new_ind], Yn[new_ind]
                weight_r = ridge_regression(x_new, y_new, lmbd)
                weight_new.append(weight_r)
            return weight_new

        def find_significant_w(arr_mean, arr_std):
            p_values = 2 * (1 - norm.cdf(np.abs(arr_mean / arr_std)))
            significant_weights = np.where(p_values < 0.05)
            return significant_weights

        weight_new = []
        weight_new = bootstrap(500, Xn, Yn, 0)

        weight_new_mean = np.mean(weight_new, axis=0)
        weight_new_std = np.std(weight_new, axis=0)
        plt.figure(figsize=(20, 10))
        plt.grid()
        plt.errorbar(np.arange(1, 101), weight_new_mean, yerr=2 * weight_new_std, ecolor='r', fmt='o-k', capsize=5)
        plt.ylabel(r'Resampled Weight Values')
        plt.xlabel(r'Weight Indices')
        plt.title(r'Ridge Regression with ' r'$\lambda = 0$''\nand %95 CI')
        plt.show(block=False)
        print("Indices of the Resampled Weights which are significantly different than zero:")
        print(find_significant_w(weight_new_mean, weight_new_std)[0])

        print("Part C")

        weight_new_ridge = []
        weight_new_ridge = bootstrap(500, Xn, Yn, lambda_opt)
        weight_newR_mean = np.mean(weight_new_ridge, axis=0)
        weight_newR_std = np.std(weight_new_ridge, axis=0)
        plt.figure(figsize=(20, 10))
        plt.grid()
        plt.errorbar(np.arange(1, 101), weight_newR_mean, yerr=2 * weight_newR_std, ecolor='r', fmt='o-k', capsize=5)
        plt.ylabel(r'Resampled Weight Values')
        plt.xlabel(r'Weight Indices')
        plt.title(r'Ridge Regression with ' r'$\lambda = \lambda_{opt}$''\nand %95 CI')
        plt.show(block=False)
        print("Indices of the Resampled Weights which are significantly different than zero:")
        print(find_significant_w(weight_newR_mean, weight_newR_std)[0])


    elif question == '2':

        print("Question 2")

        print("Part A")

        with h5py.File('hw3_data3.mat', 'r') as file:
            pop1, pop2 = np.array(list(file['pop1'])).flatten(), np.array(list(file['pop2'])).flatten()

        def bootstrap(iter_num, x, seed=6):
            np.random.seed(seed)
            x_new = []
            for i in range(iter_num):
                new_ind = np.random.choice(np.arange(np.size(x)), np.size(x))
                x_sample = x[new_ind]
                x_new.append(x_sample)
            return np.array(x_new)

        def mean_difference(x, y, iterations):
            xy_concat = np.concatenate((x, y))
            xy_boot = bootstrap(iterations, xy_concat)
            x_boot = np.zeros((iterations, np.size(x)))
            y_boot = np.zeros((iterations, np.size(y)))
            for i in range(np.size(xy_concat)):
                if i < np.size(x):
                    x_boot[:, i] = xy_boot[:, i]
                else:
                    y_boot[:, i - np.size(x)] = xy_boot[:, i]
            x_means = np.mean(x_boot, axis=1)
            y_means = np.mean(y_boot, axis=1)
            mean_diff = x_means - y_means

            return mean_diff

        mean_diff = mean_difference(pop1, pop2, 10000)

        def find_z_and_p(x, mu):
            mu_0 = np.mean(x)
            sigma = np.std(x)
            z = np.abs((mu - mu_0) / sigma)
            p = (1 - norm.cdf(z))
            return z, p

        plt.figure()
        plt.title('Population Mean Difference')
        plt.xlabel('Difference of Means')
        plt.ylabel('P(x)')
        plt.yticks([])
        plt.hist(mean_diff, bins=60, density=True, edgecolor='black')
        plt.show(block=False)

        z, p = find_z_and_p(mean_diff, np.mean(pop1) - np.mean(pop2))
        print("z-score: ", z)
        print("two sided p-value: ", 2 * p)

        print("Part B")

        with h5py.File('hw3_data3.mat', 'r') as file:
            vox1, vox2 = np.array(list(file['vox1'])).flatten(), np.array(list(file['vox2'])).flatten()

        vox1_boot = bootstrap(10000, vox1)
        vox2_boot = bootstrap(10000, vox2)

        corr_boot = np.zeros(10000)
        for i in range(10000):
            corr_boot[i] = np.corrcoef(vox1_boot[i], vox2_boot[i])[0, 1]

        corr_mean = np.mean(corr_boot)
        sorted_corr = np.sort(corr_boot)
        dif = np.size(sorted_corr) / 40
        corr_lower = sorted_corr[int(dif)]
        corr_upper = sorted_corr[int(np.size(sorted_corr) - dif)]
        print("Mean: ", corr_mean)
        print("%95 CI: (", corr_lower, ", ", corr_upper, ")")

        zero_corr = np.where(corr_boot < 10 ** (-2))
        print("Number of elements with zero correlation: ", np.size(zero_corr))

        print("Part C")

        vox1_indep = bootstrap(10000, vox1, 13)
        vox2_indep = bootstrap(10000, vox2, 5)

        corr_boot_indep = np.zeros(10000)
        for i in range(10000):
            corr_boot_indep[i] = np.corrcoef(vox1_indep[i], vox2_indep[i])[0, 1]

        plt.figure()
        plt.title('Correlation between vox1 and vox2')
        plt.xlabel('Correlation (x)')
        plt.ylabel('P(x)')
        plt.yticks([])
        plt.hist(corr_boot_indep, bins=60, density=True, edgecolor='black')
        plt.show(block=False)

        z, p = find_z_and_p(corr_boot_indep, np.corrcoef(vox1, vox2)[0, 1])
        print("z-score: ", z)
        print("one sided p value: ", p)

        print("Part D")

        with h5py.File('hw3_data3.mat', 'r') as file:
            building, face = np.array(list(file['building'])).flatten(), np.array(list(file['face'])).flatten()

        mean_diff_d = np.zeros(10000)
        diff_options = np.zeros(4)
        choices = np.zeros(20)

        for i in range(10000):
            for j in range(20):
                ind = np.random.choice(20)
                diff_options[0:1] = 0
                diff_options[2] = building[ind] - face[ind]
                diff_options[3] = -1 * diff_options[2]
                choices[j] = diff_options[np.random.choice(4)]
            mean_diff_d[i] = np.mean(choices)

        plt.figure()
        plt.title('Difference of Means\nBuilding - Face\n(Subject Population = Same)')
        plt.xlabel('Difference of Means (x)')
        plt.ylabel('P(x)')
        plt.yticks([])
        plt.hist(mean_diff_d, bins=60, density=True, edgecolor='black')
        plt.show(block=False)

        z, p = find_z_and_p(mean_diff_d, np.mean(building) - np.mean(face))
        print("z-score: ", z)
        print("Two sided p value: ", 2 * p)

        print("Part E")

        mean_diff_e = mean_difference(building, face, 10000)

        plt.figure()
        plt.title('Difference of Means\nBuilding - Face\n(Subject Population = Different)')
        plt.xlabel('Difference of Means (x)')
        plt.ylabel('P(x)')
        plt.yticks([])
        plt.hist(mean_diff_e, bins=60, density=True, edgecolor='black')
        plt.show(block=False)

        z_e, p_e = find_z_and_p(mean_diff_e, np.mean(building) - np.mean(face))
        print("z-score: ", z_e)
        print("Two sided p value: ", 2 * p_e)



ahmet_ali_nuhoglu_21602149_hw2(question)
