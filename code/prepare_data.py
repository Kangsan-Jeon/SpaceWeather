from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from public_fits_image import img_scale
from scipy.misc import imsave, imresize
import os
from tqdm import trange
from datetime import timedelta, datetime
import shutil

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
    for i in trange(len(file_list)):
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

def change_datetime(string):
    if len(string)==10:
        string='0'+string

    if string.count('-Jan-') != 0:
        string = string[7:]+'01'+string[0:2]
    elif string.count('-Feb-') != 0:
        string = string[7:]+'02'+string[0:2]
    elif string.count('-Mar-') != 0:
        string = string[7:] + '03' + string[0:2]
    elif string.count('-Apr-') != 0:
        string = string[7:] + '04' + string[0:2]
    elif string.count('-May-') != 0:
        string = string[7:] + '05' + string[0:2]
    elif string.count('-Jun-') != 0:
        string = string[7:] + '06' + string[0:2]
    elif string.count('-Jul-') != 0:
        string = string[7:] + '07' + string[0:2]
    elif string.count('-Aug-') != 0:
        string = string[7:] + '08' + string[0:2]
    elif string.count('-Sep-') != 0:
        string = string[7:] + '09' + string[0:2]
    elif string.count('-Oct-') != 0:
        string = string[7:] + '10' + string[0:2]
    elif string.count('-Nov-') != 0:
        string = string[7:] + '11' + string[0:2]
    elif string.count('-Dec-') != 0:
        string = string[7:] + '12' + string[0:2]
    return string

def convertLabel(label_dir, output_name, flare_classes):
    '''
    # Flare event listing
    # https://hesperia.gsfc.nasa.gov/goes/goes_event_listings/

    :param label_dir: directory path of label text file ex) C:/home/
    :param output_name: output text file name ex) output.txt
    :param output_name: flare classes to want ex) ["C", "M", "X"]
    :return:
    '''
    label_list = os.listdir(label_dir)
    f_out = open(label_dir + output_name, 'w')
    for file in label_list:
        f_in = open(label_dir + file, 'r')
        for i in range(6):
            f_in.readline()
        for line in f_in:
            line = line.split()
            if len(line) != 0:
                line[0] = change_datetime(line[0])
                if (line[4][0] in flare_classes):
                    f_out.write('%s %s %s\n' % (line[0], line[1], line[4][0]))
            else:
                continue
        f_in.close()
    f_out.close()
    return

def classifyImage(image_dir, save_dir, label_file, time_duration):
    '''
    :param image_dir: image directory to classify ex) C:/home/
    :param save_dir: save directory ex) C:/home/
    :param label_file: label text file to refer
    :param time_duration: time duration(hours) between image and flare
    :return:
    '''
    none_dir = save_dir + "none/"
    c_dir = save_dir + "C/"
    m_dir = save_dir + "M/"
    x_dir = save_dir + "X/"
    image_file = os.listdir(image_dir)
    classes = []
    dt = timedelta(hours=time_duration)
    for i in range(trange(len(image_file))):
        classes.clear()
        image_name = image_file[i]
        image_time = datetime(year=int(image_name[8:12]), month=int(image_name[12:14]),
                            day=int(image_name[14:16]), hour=int(image_name[17:19]),
                            minute=int(image_name[19:21]))
        label_txt = open(label_file)
        for line in label_txt:
            flare_time = datetime(year=int(line[:4]), month=int(line[4:6]),
                                  day=int(line[6:8]), hour=int(line[9:11]),
                                  minute=int(line[12:14]))
            if (flare_time >= image_time and flare_time <= (image_time + dt)):
                classes.append(line[15:16])
            elif (flare_time > image_time + dt):
                break

        if (classes.count('X') != 0):
            shutil.copy(image_dir + image_name, x_dir + image_name)
        elif (classes.count('M') != 0):
            shutil.copy(image_dir + image_name, m_dir + image_name)
        elif (classes.count('C') != 0):
            shutil.copy(image_dir + image_name, c_dir + image_name)
        else:
            shutil.copy(image_dir + image_name, none_dir + image_name)
    return