#!/usr/bin/env python

import numpy
from astropy.io import fits
import img_scale
import matplotlib.pyplot as plt

fn = "E:/EIT/EIT_284/2002/EIT_284_20020523_070558.fits"
sig_fract = 5.0
percent_fract = 0.01

hdulist = fits.open(fn)
img_header = hdulist[0].header
img_data_raw = hdulist[0].data
hdulist.close()

width=img_data_raw.shape[0]
height=img_data_raw.shape[1]
print("#INFO : ", fn, width, height)

img_data_raw = numpy.array(img_data_raw, dtype=float)
#sky, num_iter = img_scale.sky_median_sig_clip(img_data, sig_fract, percent_fract, max_iter=100)
sky, num_iter = img_scale.sky_mean_sig_clip(img_data_raw, sig_fract, percent_fract, max_iter=10)
print("sky = ", sky, '(', num_iter, ')')
#img_data = img_data_raw - sky
img_data = numpy.clip(img_data_raw, .8e3, 1.5e3)
min_val = 0.
print("... min. and max. value : ", numpy.min(img_data), numpy.max(img_data))

# SQRT Scale
img_data = numpy.clip(img_data_raw, .8e3, 1.5e3)

fig = plt.figure()
fig.set_size_inches(width/height, 1, forward=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
new_img = img_scale.sqrt(img_data, scale_min = min_val)
ax.imshow(new_img, interpolation='nearest', origin='lower', cmap="gray")
plt.axis('off')
plt.savefig('sqrt.png', dpi = height)
plt.clf()

# Power Scale
img_data = numpy.clip(img_data_raw, .8e3, 1.5e3)

fig = plt.figure()
fig.set_size_inches(width/height, 1, forward=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
new_img = img_scale.power(img_data, power_index=3.0, scale_min = min_val)
ax.imshow(new_img, interpolation='nearest', origin='lower', cmap="gray")
plt.axis('off')
plt.savefig('power.png', dpi = height)
plt.clf()


# Log Scale
img_data = numpy.clip(img_data_raw, .8e3, 1.e4)

fig = plt.figure()
fig.set_size_inches(width/height, 1, forward=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
new_img = img_scale.log(img_data, scale_min = min_val)
ax.imshow(new_img, interpolation='nearest', origin='lower', cmap="gray")
plt.axis('off')
plt.savefig('log.png', dpi = height)
plt.clf()

fig = plt.figure()
fig.set_size_inches(width/height, 1, forward=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
new_img = img_scale.linear(img_data, scale_min = min_val)
ax.imshow(new_img, interpolation='nearest', origin='lower', cmap="gray")
plt.savefig('linear.png', dpi = height)
plt.clf()

'''
new_img = img_scale.asinh(img_data, scale_min = min_val, non_linear=0.01)
plt.imshow(new_img, interpolation='nearest', origin='lower', cmap="gray")
plt.axis('off')
plt.savefig('asinh_beta_01.png')
plt.clf()
new_img = img_scale.asinh(img_data, scale_min = min_val, non_linear=0.5)
plt.imshow(new_img, interpolation='nearest', origin='lower', cmap="gray")
plt.axis('off')
plt.savefig('asinh_beta_05.png')
plt.clf()

new_img = img_scale.asinh(img_data, scale_min = min_val, non_linear=2.0)
plt.imshow(new_img, interpolation='nearest', origin='lower', cmap="gray")
plt.axis('off')
plt.savefig('asinh_beta_20.png')
plt.clf()

new_img = img_scale.histeq(img_data_raw, num_bins=256)
plt.imshow(new_img, interpolation='nearest', origin='lower', cmap="gray")
plt.axis('off')
plt.savefig('histeq.png')
plt.clf()

new_img = img_scale.logistic(img_data_raw, center = 0.03, slope = 0.3)
plt.imshow(new_img, interpolation='nearest', origin='lower', cmap="gray")
plt.axis('off')
plt.savefig('logistic.png')
plt.clf()
'''