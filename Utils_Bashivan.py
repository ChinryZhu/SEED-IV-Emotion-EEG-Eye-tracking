'''
Created by Pouya bashivan
This code has been created by p. bashivan source : https://github.com/pbashivan/EEGLearn
'''

__author__ = 'Pouya Bashivan'

import numpy as np
from scipy.interpolate import griddata
from sklearn.preprocessing import scale
import math as m

def gen_images(locs_path, features, n_gridpoints, normalize=True,
               edgeless=False):

    feat_array_temp = []
    n_samples,n_features,n_colors=features.shape

    for c in range(n_colors):
        feat_array_temp.append(features[:, :, c])
    locs=np.load(locs_path)

    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                     ]
    temp_interp = [np.zeros([n_samples, n_gridpoints, n_gridpoints]) for _ in range(n_colors)]

    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.concatenate([feat_array_temp[c], np.zeros((n_samples, 4))], axis=1)

    # Interpolating
    for i in range(n_samples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                               method='cubic', fill_value=np.nan)
        print('Interpolating {0}/{1}\r'.format(i + 1, n_samples), end='\r')

    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])
    return np.array(temp_interp).transpose(1, 0, 2, 3)     # swap axes to have [samples, colors, W, H]


def azim_proj(pos):
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)
#Convert Cartesian coordinates (x, y, z) into two-dimensional projection coordinates (x_proj, y_proj)
#The algorithms used in the process are defined as follows

def cart2sph(x, y, z):
    x2_y2 = x**2 + y**2
    r = m.sqrt(x2_y2 + z**2)                    # r
    elev = m.atan2(z, m.sqrt(x2_y2))            # Elevation
    az = m.atan2(y, x)                          # Azimuth
    return r, elev, az
#Convert three-dimensional Cartesian coordinates (x, y, z) to (radius, elevation Angle, azimuth Angle) in the spherical coordinate system

def pol2cart(theta, rho):
    return rho * m.cos(theta), rho * m.sin(theta)
#Convert the points (θ, ρ) in the polar coordinate system to the points (x, y) in the Cartesian coordinate system
