from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import pprint
import os, collections
from PIL import Image

path_dir = 'E:/EIT/EIT_284/2006/'

for i in range(2000, 2012):
    path_dir = 'E:/EIT/EIT_284/{}/'.format(i)
    file_list = os.listdir(path_dir)
    for file in file_list:
        if file.find('0711_13') is not -1:
            print(file)
            image_file = fits.open(path_dir + file)
            image_data = image_file[0]
            image_data = image_data.data
            clip_image = np.clip(image_data, .8e3, 1.5e3)
            lst = image_data.flatten()
            m = mode(lst)
            print("Min: ", np.min(lst))
            print("Max: ", np.max(lst))
            print("Mean: ", np.mean(lst))
            print("Mode: ", m[0][0], ', # of mode value: ', np.count_nonzero(lst == m[0][0]))
            # plot histogram
            plt.figure("Histogram")
            histogram = plt.hist(image_data.flatten(), label='{}'.format(file[8:16]),
                                 bins=1000, range=[.8e3, 1.5e3])
            plt.legend()

            # plot image file
            # plt.figure("Image")
            # plt.imshow(image_data[:, :], cmap='gray', vmin=.8e3, vmax=1.2e3)
            # plt.figure("Clipped Image")
            # plt.axis('off')
            # plt.imshow(clip_image[:, :], cmap='gray')

'''
file_list = os.listdir(path_dir)
#print(file_list)
for file in file_list:
    if file.find('EIT_284_20060104_130609') is not -1:
        # print(file)
        image_file = fits.open(path_dir + file)
        image_data = image_file[0]
        image_data = image_data.data

        # read image data
        # print(image_data)
        #for i in image_data:
        #    print(i, max(i), min(i))
        # print("Length: ", len(image_data.flatten()))
        clip_image = np.clip(image_data, .9e3, 1.5e3)
        print("Mean: ", np.mean(image_data.flatten()))
        print("Min: ", np.min(image_data.flatten()))
        print("Max: ", np.max(image_data.flatten()))
        print(image_file[0].header['EXPMODE'])

        # plot histogram
        plt.figure("Histogram")
        histogram = plt.hist(image_data.flatten(), label='{}'.format(file), bins=1000, range=[1.3e3, 1.5e4])
        plt.legend()

        # plot image file
        plt.figure("Image")
        plt.imshow(image_data[:,:], cmap='gray', vmin=.9e3, vmax=1.5e4)
        plt.figure("Clipped Image")
        plt.axis('off')
        plt.imshow(clip_image[:, :], cmap = 'gray')
'''
plt.show()
