from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import pims
import trackpy as tp


'''the purpose of this script is to implement the trackpy package and some of its functionality on the segmented data
that comes from python'''

mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')

#frames = pims.TiffStack('/Users/apple/Documents/Research/NiCr XPEEM/segmentation test movie 3/light region try 1 segmented.tif')
frames = pims.TiffStack('multipage.tif')

f = tp.batch(frames[:], 5, processes= 1)
#f = tp.locate(frames[290], 5)
# plt.figure()
# tp.annotate(f, frames[290])

t = tp.link_df(f, 5, memory=3)
t1 = tp.filter_stubs(t, 20)

print('Before:', t['particle'].nunique())
print('After:', t1['particle'].nunique())

plt.figure()
tp.mass_size(t1.groupby('particle').mean())  # convenience function -- just plots size vs. mass

plt.figure()
tp.plot_traj(t1)
