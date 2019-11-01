from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from public_fits_image import img_scale
from scipy.misc import imsave, imresize
import os
import tqdm

def readFits(file_name, clip_min_value, clip_max_value):
    '''
    :param file_name: ex) 'E:/EIT/EIT_284/2010/image_name'
    :param min_value: min value of FITS file to clip ex) .8e3
    :param max_value: max value of FITS file to clip ex) 1.5e3
    :return:
    '''
    try:
        image = fits.open(file_name)
        image_data = image[0].data
        clip_image = np.clip(image_data, clip_min_value, clip_max_value)
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

    return

def savePngFromFits(fits_dir, png_dir, scale="power", clip_min_value=.8e3, clip_max_value=7.e3):
    '''
    :param fits_dir: original fits file directory ex) C:/home/
    :param png_dir: png file directory to save ex) C:/home/
    :param scale: convert scale(linear, power, log, sqrt)
    :param clip_min_value: min value of FITS file to clip ex) .8e3
    :param clip_max_value: max value of FITS file to clip ex) 7 .e3
    :return:
    '''
    save_dir = png_dir + "/" + scale
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file_list = os.listdir(fits_dir)
    for i in tqdm.trange(len(file_list)):
        hdulist = fits.open(fits_dir + file_list[i])
        img_header = hdulist[0].header
        try:
            img_data_raw = hdulist[0].data
            hdulist.close()
            img_data_raw = np.array(img_data_raw, dtype=float)
            img_data = np.clip(img_data_raw, clip_min_value, clip_max_value)
            if scale == "power":
                convert_img = img_scale.power(img_data, scale_min=0.)
            elif scale == "linear":
                convert_img = img_scale.linear(img_data, scale_min=0.)
            elif scale == "log":
                convert_img = img_scale.log(img_data, scale_min=0.)
            elif scale == "sqrt":
                convert_img = img_scale.sqrt(img_data, scale_min=0.)
            else:
                print("There is no {} to convert".format(scale))
                return
            downsampled_img = imresize(convert_img, [224, 224])
            imsave(save_dir + "/" + "{}.png".format(file_list[i].split(".")[0]), downsampled_img)
        except:
            continue
    return



