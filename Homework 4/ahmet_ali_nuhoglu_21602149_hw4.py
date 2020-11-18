# Ahmet Ali Nuhoğlu
# 21602149

# EEE482 - Computational Neuroscience Homework 4

import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF

question = sys.argv[1]

def ahmet_ali_nuhoglu_21602149_hw4(question):
    if question == '1' :

        print("Question 1")
        print("Part A")
        hw4_data1 = dict()
        with h5py.File('hw4_data1.mat', 'r') as file:
            hw4_data1['faces'] = np.array(file['faces'])
            print(np.shape(hw4_data1['faces']))
        faces = hw4_data1['faces'].T

        # some sample images
        for i in range(0, 5):
            plt.figure()
            plt.imshow(faces[i].reshape(32, 32).T, cmap=plt.cm.gray)
            plt.title('Image Sample %s\n32x32' % str(i + 1))
            plt.show(block=False)

        faces_pca = PCA(100)
        faces_pca.fit(faces)

        plt.figure()
        plt.plot(faces_pca.explained_variance_ratio_)
        plt.xlabel('Principal Component Index')
        plt.ylabel('Proportion with Respect to Total Variance')
        plt.title('Contribution of each Principal Component \n to Total Variance')
        plt.grid()
        plt.show(block=False)

        fig, axes = plt.subplots(5, 5, figsize=(10, 10), facecolor='white', subplot_kw={'xticks': [], 'yticks': []})
        # fig.tight_layout()
        fig.suptitle('Principal Components of the \n First 25 Images', fontsize='16')
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=.3, wspace=-.4)
        for i, ax in enumerate(axes.flat):
            ax.imshow(faces_pca.components_[i].reshape(32, 32).T, cmap=plt.cm.gray)
            ax.set_xlabel(i + 1)

        print("Part B")

        plt.figure()
        faces_pca_mean = faces_pca.mean_
        plt.imshow(faces_pca_mean.reshape(32, 32).T, cmap=plt.cm.gray)
        plt.title("Mean Image of all PCs")
        plt.show(block=False)

        fig, axes = plt.subplots(6, 6, figsize=(10, 10), facecolor='white', subplot_kw={'xticks': [], 'yticks': []})
        fig.suptitle('Original Versions of the First 36 Images', fontsize='16')
        fig.tight_layout(rect=[0, 0, 1, .95])
        for i, ax in enumerate(axes.flat):
            ax.imshow(faces[i].reshape(32, 32).T, cmap=plt.cm.gray)
            ax.set_xlabel(i + 1)

        faces_pca_10 = (faces - faces_pca_mean).dot(faces_pca.components_[0:10].T).dot(
            faces_pca.components_[0:10]) + faces_pca_mean  # Change if possible
        fig, axes = plt.subplots(6, 6, figsize=(10, 10), facecolor='white', subplot_kw={'xticks': [], 'yticks': []})
        fig.tight_layout(rect=[0, 0, 1, .93])
        fig.suptitle('First 36 Images Reconstructed Using\n First 10 Principal Components', fontsize='16')
        for i, ax in enumerate(axes.flat):
            ax.imshow(faces_pca_10[i].reshape(32, 32).T, cmap=plt.cm.gray)
            ax.set_xlabel(i + 1)

        pca_10_mse = (faces_pca_10 - faces) ** 2
        print("Mean of MSE: %f" % np.mean(pca_10_mse))
        print("Standard Deviation of MSE: %f" % np.std(np.mean(pca_10_mse, axis=1)))

        faces_pca_25 = (faces - faces_pca_mean).dot(faces_pca.components_[0:25].T).dot(
            faces_pca.components_[0:25]) + faces_pca_mean  # Change if possible
        fig, axes = plt.subplots(6, 6, figsize=(10, 10), facecolor='white', subplot_kw={'xticks': [], 'yticks': []})
        fig.tight_layout(rect=[0, 0, 1, .93])
        fig.suptitle('First 36 Images Reconstructed Using\n First 25 Principal Components', fontsize='16')
        for i, ax in enumerate(axes.flat):
            ax.imshow(faces_pca_25[i].reshape(32, 32).T, cmap=plt.cm.gray)
            ax.set_xlabel(i + 1)

        pca_25_mse = (faces_pca_25 - faces) ** 2
        print("Mean of MSE: %f" % np.mean(pca_25_mse))
        print("Standard Deviation of MSE: %f" % np.std(np.mean(pca_25_mse, axis=1)))

        faces_pca_50 = (faces - faces_pca_mean).dot(faces_pca.components_[0:50].T).dot(
            faces_pca.components_[0:50]) + faces_pca_mean  # Change if possible
        fig, axes = plt.subplots(6, 6, figsize=(10, 10), facecolor='white', subplot_kw={'xticks': [], 'yticks': []})
        fig.tight_layout(rect=[0, 0, 1, .93])
        fig.suptitle('First 36 Images Reconstructed Using\n First 50 Principal Components', fontsize='16')
        for i, ax in enumerate(axes.flat):
            ax.imshow(faces_pca_50[i].reshape(32, 32).T, cmap=plt.cm.gray)
            ax.set_xlabel(i + 1)

        pca_50_mse = (faces_pca_50 - faces) ** 2
        print("Mean of MSE: %f" % np.mean(pca_50_mse))
        print("Standard Deviation of MSE: %f" % np.std(np.mean(pca_50_mse, axis=1)))

        print("Part C")

        ica_10 = FastICA(10, random_state=np.random.seed(2)).fit(faces)

        fig, axes = plt.subplots(2, 5, figsize=(10, 5), facecolor='white', subplot_kw={'xticks': [], 'yticks': []})
        fig.tight_layout(rect=[0, 0, 1, .93])
        fig.suptitle('10 Independent Components', fontsize='16')
        for i, ax in enumerate(axes.flat):
            ax.imshow(ica_10.components_[i].reshape(32, 32).T, cmap=plt.cm.gray)
            ax.set_xlabel(i + 1)

        ica_25 = FastICA(25, random_state=np.random.seed(2)).fit(faces)

        fig, axes = plt.subplots(5, 5, figsize=(10, 10), facecolor='white', subplot_kw={'xticks': [], 'yticks': []})
        fig.tight_layout(rect=[0, 0, 1, .93])
        fig.suptitle('25 Independent Components', fontsize='16')
        for i, ax in enumerate(axes.flat):
            ax.imshow(ica_25.components_[i].reshape(32, 32).T, cmap=plt.cm.gray)
            ax.set_xlabel(i + 1)

        ica_50 = FastICA(50, random_state=np.random.seed(2)).fit(faces)
        fig, axes = plt.subplots(5, 10, figsize=(20, 10), facecolor='white', subplot_kw={'xticks': [], 'yticks': []})
        fig.tight_layout(rect=[0, 0, 1, .96])
        fig.suptitle('50 Independent Components', fontsize='16')
        for i, ax in enumerate(axes.flat):
            ax.imshow(ica_50.components_[i].reshape(32, 32).T, cmap=plt.cm.gray)
            ax.set_xlabel(i + 1)

        ica_10_reconstructed = ica_10.fit(faces).transform(faces).dot(ica_10.mixing_.T) + ica_10.mean_

        fig, axes = plt.subplots(6, 6, figsize=(10, 10), facecolor='white', subplot_kw={'xticks': [], 'yticks': []})
        fig.tight_layout(rect=[0, 0, 1, .93])
        fig.suptitle('First 36 Images Reconstructed Using\n 10 Independent Components', fontsize='16')
        for i, ax in enumerate(axes.flat):
            ax.imshow(ica_10_reconstructed[i].reshape(32, 32).T, cmap=plt.cm.gray)
            ax.set_xlabel(i + 1)

        ica_10_mse = (ica_10_reconstructed - faces) ** 2
        print("Mean of MSE: %f" % np.mean(ica_10_mse))
        print("Standard Deviation of MSE: %f" % np.std(np.mean(ica_10_mse, axis=1)))

        ica_25_reconstructed = ica_25.fit(faces).transform(faces).dot(ica_25.mixing_.T) + ica_25.mean_

        fig, axes = plt.subplots(6, 6, figsize=(10, 10), facecolor='white', subplot_kw={'xticks': [], 'yticks': []})
        fig.tight_layout(rect=[0, 0, 1, .93])
        fig.suptitle('First 36 Images Reconstructed Using\n 25 Independent Components', fontsize='16')
        for i, ax in enumerate(axes.flat):
            ax.imshow(ica_25_reconstructed[i].reshape(32, 32).T, cmap=plt.cm.gray)
            ax.set_xlabel(i + 1)

        ica_25_mse = (ica_25_reconstructed - faces) ** 2
        print("Mean of MSE: %f" % np.mean(ica_25_mse))
        print("Standard Deviation of MSE: %f" % np.std(np.mean(ica_25_mse, axis=1)))

        ica_50_reconstructed = ica_50.fit(faces).transform(faces).dot(ica_50.mixing_.T) + ica_50.mean_

        fig, axes = plt.subplots(6, 6, figsize=(10, 10), facecolor='white', subplot_kw={'xticks': [], 'yticks': []})
        fig.tight_layout(rect=[0, 0, 1, .93])
        fig.suptitle('First 36 Images Reconstructed Using\n 50 Independent Components', fontsize='16')
        for i, ax in enumerate(axes.flat):
            ax.imshow(ica_50_reconstructed[i].reshape(32, 32).T, cmap=plt.cm.gray)
            ax.set_xlabel(i + 1)

        ica_50_mse = (ica_50_reconstructed - faces) ** 2
        print("Mean of MSE: %f" % np.mean(ica_50_mse))
        print("Standard Deviation of MSE: %f" % np.std(np.mean(ica_50_mse, axis=1)))

        print("Part D")

        nnmf_faces = faces + np.abs(np.min(faces))
        nnmf_10 = NMF(n_components=10, solver="mu", max_iter=500)
        nnmf_10_w = nnmf_10.fit(nnmf_faces).transform(nnmf_faces)

        fig, axes = plt.subplots(2, 5, figsize=(10, 5), facecolor='white', subplot_kw={'xticks': [], 'yticks': []})
        fig.tight_layout(rect=[0, 0, 1, .93])
        fig.suptitle('10 MFs', fontsize='16')
        for i, ax in enumerate(axes.flat):
            ax.imshow(nnmf_10.components_[i].reshape(32, 32).T, cmap=plt.cm.gray)
            ax.set_xlabel(i + 1)

        nnmf_25 = NMF(n_components=25, solver="mu", max_iter=1000)
        nnmf_25_w = nnmf_25.fit(nnmf_faces).transform(nnmf_faces)

        fig, axes = plt.subplots(5, 5, figsize=(10, 10), facecolor='white', subplot_kw={'xticks': [], 'yticks': []})
        fig.tight_layout(rect=[0, 0, 1, .93])
        fig.suptitle('25 MFs', fontsize='16')
        for i, ax in enumerate(axes.flat):
            ax.imshow(nnmf_25.components_[i].reshape(32, 32).T, cmap=plt.cm.gray)
            ax.set_xlabel(i + 1)

        nnmf_faces = faces + np.abs(np.min(faces))
        nnmf_50 = NMF(n_components=50, solver="mu", max_iter=500)
        nnmf_50_w = nnmf_50.fit(nnmf_faces).transform(nnmf_faces)

        fig, axes = plt.subplots(5, 10, figsize=(20, 10), facecolor='white', subplot_kw={'xticks': [], 'yticks': []})
        fig.tight_layout(rect=[0, 0, 1, .96])
        fig.suptitle('50 MFs', fontsize='16')
        for i, ax in enumerate(axes.flat):
            ax.imshow(nnmf_50.components_[i].reshape(32, 32).T, cmap=plt.cm.gray)
            ax.set_xlabel(i + 1)

        nnmf_10_reconstructed = nnmf_10_w.dot(nnmf_10.components_) - np.abs(np.min(faces))

        fig, axes = plt.subplots(6, 6, figsize=(10, 10), facecolor='white', subplot_kw={'xticks': [], 'yticks': []})
        fig.tight_layout(rect=[0, 0, 1, .93])
        fig.suptitle('First 36 Images Reconstructed Using\n 10 MFs', fontsize='16')
        for i, ax in enumerate(axes.flat):
            ax.imshow(nnmf_10_reconstructed[i].reshape(32, 32).T, cmap=plt.cm.gray)
            ax.set_xlabel(i + 1)

        nnmf_10_mse = (nnmf_10_reconstructed - faces) ** 2
        print("Mean of MSE: %f" % np.mean(nnmf_10_mse))
        print("Standard Deviation of MSE: %f" % np.std(np.mean(nnmf_10_mse, axis=1)))

        nnmf_25_reconstructed = nnmf_25_w.dot(nnmf_25.components_) - np.abs(np.min(faces))

        fig, axes = plt.subplots(6, 6, figsize=(10, 10), facecolor='white', subplot_kw={'xticks': [], 'yticks': []})
        fig.tight_layout(rect=[0, 0, 1, .93])
        fig.suptitle('First 36 Images Reconstructed Using\n 25 MFs', fontsize='16')
        for i, ax in enumerate(axes.flat):
            ax.imshow(nnmf_25_reconstructed[i].reshape(32, 32).T, cmap=plt.cm.gray)
            ax.set_xlabel(i + 1)

        nnmf_25_mse = (nnmf_25_reconstructed - faces) ** 2
        print("Mean of MSE: %f" % np.mean(nnmf_25_mse))
        print("Standard Deviation of MSE: %f" % np.std(np.mean(nnmf_25_mse, axis=1)))

        nnmf_50_reconstructed = nnmf_50_w.dot(nnmf_50.components_) - np.abs(np.min(faces))

        fig, axes = plt.subplots(6, 6, figsize=(10, 10), facecolor='white', subplot_kw={'xticks': [], 'yticks': []})
        fig.tight_layout(rect=[0, 0, 1, .93])
        fig.suptitle('First 36 Images Reconstructed Using\n 50 MFs', fontsize='16')
        for i, ax in enumerate(axes.flat):
            ax.imshow(nnmf_50_reconstructed[i].reshape(32, 32).T, cmap=plt.cm.gray)
            ax.set_xlabel(i + 1)

        nnmf_50_mse = (nnmf_50_reconstructed - faces) ** 2
        print("Mean of MSE: %f" % np.mean(nnmf_50_mse))
        print("Standard Deviation of MSE: %f" % np.std(np.mean(nnmf_50_mse, axis=1)))


    elif question == '2':

        print("Question 2")

        print("Part A")

        mu_values = np.arange(-10, 11, 1)

        def tuning_curves(x, mu, sigma=1):
            return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        stimulus = np.linspace(-20, 21, 1000)

        plt.figure(figsize=(20, 10))
        for mu in mu_values:
            plt.plot(stimulus, tuning_curves(stimulus, mu))
        plt.minorticks_on()
        plt.grid(which='both')
        plt.xlabel('Stimulus')
        plt.ylabel('Responses')
        plt.title('All Tuning Curves')
        plt.show(block=False)

        plt.figure(figsize=(20, 10))
        plt.plot(mu_values, tuning_curves(-1, mu_values))
        plt.minorticks_on()
        plt.grid(which='both')
        plt.xlabel('Prefered Stimulus')
        plt.ylabel('Population Response')
        plt.title('Each Neuorn\'s Prefered Stimulus Value')
        plt.show(block=False)

        print("Part B")

        stimulus_interval = np.linspace(-5, 5, 1000)

        def wta_decoder(pref_stim, response):
            return pref_stim[np.argmax(response)]

        stimulus = []
        responses = []
        wta_est = []
        wta_error = []

        np.random.seed(6)
        for i in range(200):
            gaussian_noise = np.random.normal(0, 1 / 20, 21)
            stimulus.append(np.random.choice(stimulus_interval))
            response = tuning_curves(stimulus[i], mu_values)
            response_with_noise = response + gaussian_noise
            responses.append(response_with_noise)
            wta_est.append(wta_decoder(mu_values, response_with_noise))
            wta_error.append(np.abs(stimulus[i] - wta_est[i]))

        plt.figure(figsize=(20, 10))
        plt.scatter(np.arange(200), stimulus)
        plt.scatter(np.arange(200), wta_est)
        plt.xlabel('Trial')
        plt.ylabel('Stimulus')
        plt.title('Actual Values and winner-take-all Estimated Values\n Across 200 Trials')
        plt.legend(['Actual Values', 'Estimtes'], loc='upper right')
        plt.show(block=False)
        print("Winner-take-all decoder estimate error mean: %f" % np.mean(wta_error))
        print("Winner-take-all decoder estimate error standard deviation: %f" % np.std(wta_error))

        print("Part C")

        def ml_decoder(stim_interval, response, sigma=1):
            log_likelihood = []
            for s in stim_interval:
                log_l_val = 0
                for r, mu in zip(response, mu_values):
                    log_l_val += (r - tuning_curves(s, mu)) ** 2
                log_likelihood.append(log_l_val)
            return stim_interval[np.argmin(log_likelihood)]

        stimulus_mle, mle_error = [], []
        for resp, stim in zip(responses, stimulus):
            stimulus_mle.append(ml_decoder(stimulus_interval, resp))
            mle_error.append(float(np.abs(stim - stimulus_mle[len(stimulus_mle) - 1])))

        plt.figure(figsize=(20, 10))
        plt.scatter(np.arange(200), stimulus)
        plt.scatter(np.arange(200), stimulus_mle)
        plt.xlabel('Trial')
        plt.ylabel('Stimulus')
        plt.title('Actual vs. ML Estimates \n Across 200 Trials')
        plt.legend(['Actual Values', 'Estimates'], loc='upper right')
        plt.show(block=False)
        print("ML decoder estimate error mean: %f" % np.mean(mle_error))
        print("ML decoder estimate error standard deviation: %f" % np.std(mle_error))

        print("Part D")

        def map_decoder(stim_interval, response):
            log_posterior = []
            for stim in stim_interval:
                log_post_val = 0
                for r, mu in zip(response, mu_values):
                    log_post_val += (r - tuning_curves(stim, mu)) ** 2
                log_post_val = log_post_val * 200 + (stim ** 2) / 10
                log_posterior.append(log_post_val)
            return stim_interval[np.argmin(log_posterior)]

        stimulus_map, map_error = [], []
        for resp, stim in zip(responses, stimulus):
            stimulus_map.append(map_decoder(stimulus_interval, resp))
            map_error.append(float(np.abs(stim - stimulus_map[len(stimulus_map) - 1])))

        plt.figure(figsize=(20, 10))
        plt.scatter(np.arange(200), stimulus)
        plt.scatter(np.arange(200), stimulus_map)
        plt.xlabel('Trial')
        plt.ylabel('Stimulus')
        plt.title('Actual vs. MAP Estimates \n Across 200 Trials')
        plt.legend(['Actual Values', 'Estimates'], loc='upper right')
        plt.show(block=False)
        print("MAP decoder estimate error mean: %f" % np.mean(map_error))
        print("MAP decoder estimate error standard deviation: %f" % np.std(map_error))

        print("Part E")

        sigma_i = [.1, .2, .5, 1, 2, 5]
        xml_mle_err = []

        np.random.seed(3)
        for i in range(200):
            stim = np.random.choice(stimulus_interval)
            xi_err = []
            for sig in sigma_i:
                resp = tuning_curves(stim, mu_values, sig) + np.random.normal(0, 1 / 20, 21)
                xi_err.append(np.abs(stim - float(ml_decoder(stimulus_interval, resp, sig))))
            xml_mle_err.append(np.array(xi_err))

        xml_mle_err = np.array(xml_mle_err)

        print("Mean of errors in MLE (for Sigma = 0.1): %f" % np.mean(xml_mle_err[:, 0]))
        print("STD of errors in MLE (for Sigma = 0.1): %f" % np.std(xml_mle_err[:, 0]))

        print("Mean of errors in MLE (for Sigma = 0.2): %f" % np.mean(xml_mle_err[:, 1]))
        print("STD of errors in MLE (for Sigma = 0.2): %f" % np.std(xml_mle_err[:, 1]))

        print("Mean of errors in MLE (for Sigma = 0.5): %f" % np.mean(xml_mle_err[:, 2]))
        print("STD of errors in MLE (for Sigma = 0.5): %f" % np.std(xml_mle_err[:, 2]))

        print("Mean of errors in MLE (for Sigma = 1): %f" % np.mean(xml_mle_err[:, 3]))
        print("STD of errors in MLE (for Sigma = 1): %f" % np.std(xml_mle_err[:, 3]))

        print("Mean of errors in MLE (for Sigma = 2): %f" % np.mean(xml_mle_err[:, 4]))
        print("STD of errors in MLE (for Sigma = 2): %f" % np.std(xml_mle_err[:, 4]))

        print("Mean of errors in MLE (for Sigma = 5): %f" % np.mean(xml_mle_err[:, 5]))
        print("STD of errors in MLE (for Sigma = 5): %f" % np.std(xml_mle_err[:, 5]))


ahmet_ali_nuhoglu_21602149_hw4(question)
