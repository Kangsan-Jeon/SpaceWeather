from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pprint

image_file = fits.open('E:/EIT/EIT_284/1996/EIT_284_19960115_210417.fits')
image_file.info()
# image_data = fits.getdata(image_file, ext=0) ... 실행X
image_data = image_file[0]
image_data = image_data.data


# read image data
pprint.pprint(image_file[0].header)
print(image_data)
print("Length: ", len(image_data.flatten()))
print("Mean: ", np.mean(image_data.flatten()))
print("Min: ", np.min(image_data.flatten()))
print("Max: ", np.max(image_data.flatten()))

# plot histogram
plt.figure("Histogram")
histogram = plt.hist(image_data.flatten(), bins=1000, range=[.8e3,2.e3])

# plot image file
plt.figure("Image")
plt.imshow(image_data[:,:], cmap='hot', vmin=.8e3, vmax=2.e3)
plt.show()