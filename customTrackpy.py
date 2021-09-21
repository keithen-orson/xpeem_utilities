import numpy as np
import pandas as pd

import pims
import trackpy as tp
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage import morphology, util, filters
import skimage
import matplotlib.patches as mpatches
from matplotlib.pyplot import quiver
import trackpyfunctions

'''
Custom particle tracking using the trackpy algorithm.  Segmented images are loaded into this script and the regions
are found using skimage's region finding capability.  These regions are put into a dataframe that trackpy can use to 
track particles in.  


This work is based off of the tutorial http://soft-matter.github.io/trackpy/v0.3.0/tutorial/custom-feature-detection.html, 
which is derived from the work of Sauret et al. 'Damping of liquid sloshing by foams'Physics of Fluids 27, 022103 (2015); 
https://doi.org/10.1063/1.4907048
'''

mpl.rc('figure',  figsize=(6, 6))
mpl.rc('image', cmap='gray')

imagetitle = 'Dark spots 1, movie 3'
trueimage = pims.open('/Users/apple/Documents/Research/NiCr XPEEM/segmentation test movie 3/dark spots 1.tif') #this is for illustration purposes later
filepath = '/Users/apple/Documents/Research/NiCr XPEEM/Python Segmentation /'
filename = 'dark spots 1 pyseg leveled movie 3.tif'
frames = pims.open(filepath+filename)
id_example = 0
img_example = frames[id_example]


'''Create a pandas dataframe, and then find the particles (regions) using skimage's label function.  Label connects 
regions of the same integer value, i.e. segmented regions. In this dataframe, I also save the perimeter, filled fraction,
and the area.  
'''
features =trackpyfunctions.findparticles(frames)

plt.figure()
plt.title(imagetitle)
tp.annotate(features[features.frame==(id_example)], img_example)

search_range = 15    #how many pixels the particles are allowed to jump between frames
t = tp.link_df(features, search_range, memory=25)
t1 = tp.filter_stubs(t, 15)

meandata = trackpyfunctions.meanfield(t1)
plt.plot(meandata.frame, (meandata.averageradius)**3-(meandata.initialr)**3)
plt.show()

vectordata = trackpyfunctions.vectortracks(t1)

i = 0
trimmedvectordata = vectordata[vectordata.startframe == i]

plt.imshow(trueimage[i])
plt.quiver(trimmedvectordata.x0, trimmedvectordata.y0, trimmedvectordata.dx, -trimmedvectordata.dy, headwidth=4, headlength=6, color='red')
plt.show()

plt.imshow(trueimage[i])
plt.quiver(vectordata.x0, vectordata.y0, vectordata.dx, -vectordata.dy, headwidth=4, headlength=6, color='red')
plt.show()

plt.figure()
plt.imshow(trueimage[i]) #showing the frame is just for illustration, so that you can see the image overlayed
plt.title(imagetitle)
tp.plot_traj(t1)
plt.show()

#
# #compute the particle drift
#
# d = tp.compute_drift(t1)
# d.plot()
# plt.title(imagetitle)
# plt.show()
# tm = tp.subtract_drift(t1.copy(), d)
#
# em = tp.emsd(tm, .01, 1)   # microns per pixel = .01 (~10 nm, this needs to be corrected to the real value)
#                             #frames per second = 1 (this also needs to be corrected to the real frame rate
# fig, ax = plt.subplots()
# ax.plot(em.index, em, 'o')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
#        xlabel='lag time $t$')
# plt.title(imagetitle)
# plt.show()
#
# plt.figure()
# plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
# plt.xlabel('lag time $t$');
# plt.title(imagetitle)
# tp.utils.fit_powerlaw(em)  # performs linear best fit in log space, plots]
#
