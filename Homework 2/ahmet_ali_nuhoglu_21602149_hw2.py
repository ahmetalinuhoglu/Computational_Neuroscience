# Ahmet Ali Nuhoğlu
# 21602149
# EEE482 - Computational Neuroscience Homework 2

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.signal as sig
from PIL import Image
import sys

question = sys.argv[1]

def ahmet_ali_nuhoglu_21602149_hw2(question):
    if question == '1' :

        print("Question 1")
        print("Part A")

        data = sio.loadmat('c2p3.mat')
        counts = data['counts']
        stim = data['stim']
        print(np.shape(stim))
        print(np.shape(counts))

        def STA(stim, counts, time_s):
            avg = np.zeros((len(stim[0]), len(stim[1])))
            for i in range(0, len(counts) - time_s):
                avg[:, :] += stim[:, :, i] * counts[i + time_s]
            avg /= np.sum(counts[time_s:])
            return avg

        averages = np.zeros((len(stim[0]), len(stim[1]), 10))
        for i in range(0, 10):
            averages[:, :, i] = STA(stim, counts, i + 1)

        for i in range(10):
            plt.figure()
            plt.imshow(averages[:, :, i], cmap='gray', vmin=np.min(averages), vmax=np.max(averages))
            plt.colorbar()
            if i == 0:
                plt.title('STA: 1 Step Before Spike')
            else:
                plt.title('STA: %d Steps Before Spike' % (i + 1))
            plt.show(block=False)

        print("Part B")

        row_avgs = np.sum(averages, axis=0)
        plt.figure(figsize=(5, 5))
        plt.title("Row Summed Averages")
        plt.imshow(row_avgs, cmap='gray')
        plt.xticks(np.arange(0, 10, step=1))
        plt.show(block=False)

        col_avgs = np.sum(averages, axis=1)
        plt.figure(figsize=(5, 5))
        plt.title("Col Summed Averages")
        plt.imshow(col_avgs, cmap='gray')
        plt.xticks(np.arange(0, 10, step=1))
        plt.show(block=False)

        print("Part C")

        stim_proj_sta = np.zeros(len(counts))
        for i in range(len(counts)):
            for j in range(len(averages[:, 0, 0])):
                stim_proj_sta[i] += np.inner(averages[:, j, 0], stim[:, j, i])

        stim_proj_sta /= np.max(stim_proj_sta)
        plt.figure(figsize=(10, 5))
        plt.title("Stimulus Projected on STA", fontsize=15)
        plt.ylabel("Spike Counts")
        plt.xlabel("Stimulus Projection")
        plt.grid(b=1, alpha=.2)
        plt.hist(stim_proj_sta, bins=100, alpha=.75, rwidth=.66)
        plt.show()

        non_zero_index = []
        for i in range(len(counts)):
            if counts[i] != 0:
                non_zero_index.append(i)

        non_zero_proj_sta = np.zeros(len(non_zero_index))
        for i in range(len(non_zero_proj_sta)):
            for j in range(len(averages[:, 0, 0])):
                non_zero_proj_sta[i] += np.inner(averages[:, j, 0], stim[:, j, non_zero_index[i]])

        non_zero_proj_sta /= np.max(non_zero_proj_sta)
        plt.figure(figsize=(10, 5))
        plt.title("Stimulus Projected on STA for Non-Zero Spikes", fontsize=15)
        plt.ylabel("Spike Counts")
        plt.xlabel("Stimulus Projection")
        plt.grid(b=1, alpha=.2)
        plt.ylim(0, 800)
        plt.hist(non_zero_proj_sta, bins=100, alpha=0.75, color='red', rwidth=.66)
        plt.show()

        plt.figure(figsize=(10, 5))
        all_cases = plt.hist(stim_proj_sta, bins=100, label='All Cases', alpha=.7, rwidth=.66)
        non_zeros = plt.hist(non_zero_proj_sta, bins=100, color='red', label='Non-Zero Spike Cases', alpha=.7,
                             rwidth=.66)
        plt.legend()
        plt.grid(b=1, alpha=.2)
        plt.title('Comparison of the found results', fontsize=15)
        plt.ylabel("Spike Counts")
        plt.xlabel("Stimulus Projection")
        plt.show()


    elif question == '2':

        print("Question 2")

        print("Part A")

        def DOG(x, y):
            sigma_c = 2
            sigma_s = 4
            return 1 / (2 * np.pi * sigma_c ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma_c ** 2)) - 1 / (
                        2 * np.pi * sigma_s ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma_s ** 2))

        dog_receptive_field = np.zeros((21, 21))
        for i in range(-10, 11):
            for j in range(-10, 11):
                dog_receptive_field[10 + i][10 + j] = DOG(i, j)

        plt.title('21X21 DOG Receptive Field\n')
        plt.imshow(dog_receptive_field, cmap='coolwarm')
        plt.colorbar()
        plt.show()

        plt.figure(figsize=(20, 10))
        X = Y = np.linspace(-10, 10, 21)
        X, Y = np.meshgrid(X, Y)
        ax = plt.axes(projection='3d')
        ax.set_title('3D View of DOG Receptive Field\n', fontsize=25);
        ax.set_xlabel("\nX axis", fontsize=15)
        ax.set_ylabel("\nY axis", fontsize=15)
        ax.set_zlabel("\nDOG(x,y)", fontsize=15)
        ax.plot_surface(X, Y, dog_receptive_field, cmap='coolwarm', edgecolor='none')
        plt.show(block=False)

        print("Part B")

        plt.figure()
        monkey = Image.open("hw2_image.bmp")
        monkey = np.array(monkey)
        plt.imshow(monkey)
        plt.title("Given Monkey Image")
        plt.show()
        plt.figure()
        conv_img = sig.convolve(monkey[:, :, 0], dog_receptive_field, mode='same')
        plt.imshow(conv_img, cmap='gray')
        plt.title("After Filtering with DOG\nReceptive Field")
        plt.show()

        print("Part C")

        def detect_edge(img, threshold):
            result_img = np.zeros(np.shape(img))
            for i in range(np.shape(result_img)[0]):
                for j in range(np.shape(result_img)[1]):
                    if img[i, j] >= threshold:
                        result_img[i, j] = 1
                    else:
                        result_img[i, j] = 0
            return result_img

        for i in range(5):
            plt.figure()
            plt.title("Edge-Detection with DOG Filter\nThreshold = %d" % i)
            plt.imshow(detect_edge(conv_img, i), cmap='gray')


        print("Part D")

        def gabor(x, th):
            theta = th
            sigma_l = sigma_w = 3
            lmbd = 6
            phi = 0
            k_theta = np.array([np.cos(theta), np.sin(theta)])
            k_orthogonal = np.array([-np.sin(theta), np.cos(theta)])
            k_dot_x = k_theta.dot(x)
            ko_dot_x = k_orthogonal.dot(x)
            return np.exp(-(k_dot_x ** 2) / (2 * (sigma_l ** 2)) - (ko_dot_x ** 2) / (2 * (sigma_w ** 2))) * np.cos(
                2 * np.pi * ko_dot_x / lmbd + phi)

        gabor_receptive_field_90 = np.zeros((21, 21))
        for i in range(-10, 11):
            for j in range(-10, 11):
                gabor_receptive_field_90[10 + i][10 + j] = gabor(np.array([i, j]), np.pi / 2)

        plt.title('21X21 Gabor Receptive Field\n' r' with $\theta$ = 90' '\n')
        plt.imshow(gabor_receptive_field_90, cmap='coolwarm')
        plt.colorbar()
        plt.show()

        plt.figure(figsize=(20, 10))
        X = Y = np.linspace(-10, 10, 21)
        X, Y = np.meshgrid(X, Y)
        ax = plt.axes(projection='3d')
        ax.set_title('3D View of Gabor Receptive Field \n' r' with $\theta$ = 90' '\n', fontsize=25);
        ax.set_xlabel("\nX axis", fontsize=15)
        ax.set_ylabel("\nY axis", fontsize=15)
        ax.set_zlabel("\n"r"Gabor($\vec{x}$)", fontsize=15)
        ax.plot_surface(X, Y, gabor_receptive_field_90, cmap='coolwarm', edgecolor='none')
        plt.show(block=False)

        print("Part E")

        gabor_conv_90 = sig.convolve(monkey[:, :, 0], gabor_receptive_field_90, mode='same')
        plt.imshow(gabor_conv_90, cmap='gray')
        plt.title("After Filtering with Gabor\nReceptive Field with " r"$\theta$= 90")
        plt.show()

        print("Part F")

        gabor_receptive_field_0 = np.zeros((21, 21))
        for i in range(-10, 11):
            for j in range(-10, 11):
                gabor_receptive_field_0[10 + i][10 + j] = gabor(np.array([i, j]), 0)

        gabor_receptive_field_30 = np.zeros((21, 21))
        for i in range(-10, 11):
            for j in range(-10, 11):
                gabor_receptive_field_30[10 + i][10 + j] = gabor(np.array([i, j]), np.pi / 6)

        gabor_receptive_field_60 = np.zeros((21, 21))
        for i in range(-10, 11):
            for j in range(-10, 11):
                gabor_receptive_field_60[10 + i][10 + j] = gabor(np.array([i, j]), np.pi / 3)

        plt.title('21X21 Gabor Receptive Field\n' r' with $\theta$ = 0' '\n')
        plt.imshow(gabor_receptive_field_0, cmap='coolwarm')
        plt.colorbar()
        plt.show()

        gabor_conv_0 = sig.convolve(monkey[:, :, 0], gabor_receptive_field_0, mode='same')
        plt.imshow(gabor_conv_0, cmap='gray')
        plt.title("After Filtering with Gabor\nReceptive Field with " r"$\theta$= 0")
        plt.show()

        plt.title('21X21 Gabor Receptive Field\n' r' with $\theta$ = 30' '\n')
        plt.imshow(gabor_receptive_field_30, cmap='coolwarm')
        plt.colorbar()
        plt.show()

        gabor_conv_30 = sig.convolve(monkey[:, :, 0], gabor_receptive_field_30, mode='same')
        plt.imshow(gabor_conv_30, cmap='gray')
        plt.title("After Filtering with Gabor\nReceptive Field with " r"$\theta$= 30")
        plt.show()

        plt.title('21X21 Gabor Receptive Field\n' r' with $\theta$ = 60' '\n')
        plt.imshow(gabor_receptive_field_60, cmap='coolwarm')
        plt.colorbar()
        plt.show()

        gabor_conv_60 = sig.convolve(monkey[:, :, 0], gabor_receptive_field_60, mode='same')
        plt.imshow(gabor_conv_60, cmap='gray')
        plt.title("After Filtering with Gabor\nReceptive Field with " r"$\theta$= 60")
        plt.show()

        summed_image = gabor_conv_0 + gabor_conv_30 + gabor_conv_60 + gabor_conv_90
        plt.imshow(summed_image, cmap='gray')
        plt.title("Resultin Image After Summing Up\nAll Results of the Filters")
        plt.show()

        gabor_edge_0 = detect_edge(gabor_conv_0, 0)
        plt.figure()
        plt.title("Edge-Detection with Gabor Filter\nThreshold = 0 and " r"$\theta = 0$")
        plt.imshow(gabor_edge_0, cmap='gray')

        gabor_edge_30 = detect_edge(gabor_conv_30, 0)
        plt.figure()
        plt.title("Edge-Detection with Gabor Filter\nThreshold = 0 and " r"$\theta = \frac{\pi}{6}$")
        plt.imshow(gabor_edge_30, cmap='gray')

        gabor_edge_60 = detect_edge(gabor_conv_60, 0)
        plt.figure()
        plt.title("Edge-Detection with Gabor Filter\nThreshold = 0 and " r"$\theta = \frac{\pi}{3}$")
        plt.imshow(gabor_edge_60, cmap='gray')

        gabor_edge_90 = detect_edge(gabor_conv_90, 0)
        plt.figure()
        plt.title("Edge-Detection with Gabor Filter\nThreshold = 0 and " r"$\theta = \frac{\pi}{2}$")
        plt.imshow(gabor_edge_90, cmap='gray')
        plt.show(block=False)

        edge_summed_img = gabor_edge_0 + gabor_edge_30 + gabor_edge_60 + gabor_edge_90
        plt.imshow(edge_summed_img, cmap='gray')
        plt.title("Resultin Image After Summing Up\nAll Results of Edge Detection")
        plt.show()

ahmet_ali_nuhoglu_21602149_hw2(question)
