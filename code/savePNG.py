import numpy as np
from astropy.io import fits
from public_fits_image import img_scale
import os
from scipy.misc import imsave, imresize

train_path_dir = 'C:/EIT_284/train/'
val_path_dir = 'C:/EIT_284/validation/'
test_path_dir = 'C:/EIT_284/test/'

# Train_set
for i in [2009,2010,2011]:
    if not os.path.exists(train_path_dir + "sqrt_2.e3"):
        os.mkdir(train_path_dir + "sqrt_2.e3")

    file_list = os.listdir("E:/EIT/EIT_284/{}/".format(i))
    for file in file_list:
        print(file)
        min_val = 0.
        if file.find('{}'.format(i)) is not -1:
            hdulist = fits.open("E:/EIT/EIT_284/{}/".format(i) + file)
            img_header = hdulist[0].header
            try:
                img_data_raw = hdulist[0].data
                hdulist.close()
                width = img_data_raw.shape[0]
                height = img_data_raw.shape[1]
                img_data_raw = np.array(img_data_raw, dtype=float)
                img_data = np.clip(img_data_raw, .8e3, 7.e3)
                # Save
                new_img = img_scale.sqrt(img_data, scale_min=min_val)
                new_img = imresize(new_img, [224, 224])
                imsave(train_path_dir + 'sqrt_2.e3/{}.png'.format(file[0:23]), new_img)
            except:
                continue

# Validatioin_set
for k in range(2004, 2005):
    if not os.path.exists(val_path_dir + "sqrt_2.e3"):
        os.mkdir(val_path_dir + "sqrt_2.e3")

    file_list = os.listdir("E:/EIT/EIT_284/{}/".format(k))
    for file in file_list:
        print(file)
        min_val = 0.
        if file.find('{}'.format(k)) is not -1:
            hdulist = fits.open("E:/EIT/EIT_284/{}/".format(k) + file)
            img_header = hdulist[0].header
            try:
                img_data_raw = hdulist[0].data
                hdulist.close()
                width = img_data_raw.shape[0]
                height = img_data_raw.shape[1]
                img_data_raw = np.array(img_data_raw, dtype=float)
                img_data = np.clip(img_data_raw, .8e3, 7.e3)
                # Save
                new_img = img_scale.sqrt(img_data, scale_min=min_val)
                new_img = imresize(new_img, [224, 224])
                imsave(val_path_dir + 'sqrt_2.e3/{}.png'.format(file[0:23]), new_img)
            except:
                continue

# Test_set
for j in range(2005, 2006):
    if not os.path.exists(test_path_dir + "sqrt_2.e3"):
        os.mkdir(test_path_dir + "sqrt_2.e3")

    file_list = os.listdir("E:/EIT/EIT_284/{}/".format(j))
    for file in file_list:
        print(file)
        min_val = 0.
        if file.find('{}'.format(j)) is not -1:
            hdulist = fits.open("E:/EIT/EIT_284/{}/".format(j) + file)
            img_header = hdulist[0].header
            try:
                img_data_raw = hdulist[0].data
                hdulist.close()
                width = img_data_raw.shape[0]
                height = img_data_raw.shape[1]
                img_data_raw = np.array(img_data_raw, dtype=float)
                img_data = np.clip(img_data_raw, .8e3, 7.e3)
                # Save
                new_img = img_scale.sqrt(img_data, scale_min=min_val)
                new_img = imresize(new_img, [224, 224])
                imsave(test_path_dir + 'sqrt_2.e3/{}.png'.format(file[0:23]), new_img)
            except:
                continue
'''    
for file in file_list:
    print(file)
    min_val = 0.
    if file.find('20000101') is not -1:
        hdulist = fits.open("E:/EIT/EIT_284/2000/"+file)
        img_header = hdulist[0].header
        try:
            img_data_raw = hdulist[0].data
            hdulist.close()
            width = img_data_raw.shape[0]
            height = img_data_raw.shape[1]
            img_data_raw = np.array(img_data_raw, dtype=float)
            img_data = np.clip(img_data_raw, .8e3, 1.25e3)
            # Save
            fig = plt.figure()
            fig.set_size_inches(width / height, 1, forward=False)
            ax = plt.Axes(fig, [0., 0., 1.2, 1.2])
            ax.set_axis_off()
            fig.add_axes(ax)
            new_img = img_scale.log(img_data, scale_min=min_val)
            new_img = imresize(new_img, [224, 224])
            imsave(path_dir+'log/{}.png'.format(file[0:23]), new_img)
            # im = plt.imread(path_dir+'log/{}.png'.format(file[0:23]))
            # print(np.shape(im))
            # ax.imshow(new_img[:,:,1], interpolation='nearest', origin='lower', cmap='gray')
            # plt.axis('off')
            # plt.savefig(path_dir+'log/{}.png'.format(file[0:23]), dpi=224)
            # plt.clf()
            # plt.close()
        except:
            continue
            '''


