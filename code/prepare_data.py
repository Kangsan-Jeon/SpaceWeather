from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import os

def readFits(file_name, min_value, max_value):
    '''
    :param file_name: ex) 'E:/EIT/EIT_284/2010/image_name'
    :param min_value: min value of FITS file to clip ex) .8e3
    :param max_value: max value of FITS file to clip ex) 1.5e3
    :return:
    '''
    try:
        image = fits.open(file_name)
        image_data = image[0].data
        clip_image = np.clip(image_data, min_value, max_value)
        image_value = image_data.flatten()
        m = mode(image_value)
        print("Min: ", np.min(image_value))
        print("Max: ", np.max(image_value))
        print("Mean: ", np.mean(image_value))
        print("Mode: ", m[0][0], ', # of mode value: ', np.count_nonzero(image_value == m[0][0]))
        # plot histogram
        plt.figure("Histogram")
        histogram = plt.hist(image_data.flatten(), label='{}'.format(file_name),
                             bins=1000, range=[.8e3, 1.5e3])
        plt.legend()

        # plot image file
        plt.figure("Image")
        plt.imshow(image_data[:, :], cmap='gray', vmin=.8e3, vmax=1.2e3)
        plt.figure("Clipped Image")
        plt.axis('off')
        plt.imshow(clip_image[:, :], cmap='gray')
        plt.show()
    except:
        print("There is no file")