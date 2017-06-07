"""
This is a python implementation of Peter
Kovesi's colormap generation methods described
here:

http://peterkovesi.com/projects/colourmaps/

CET Perceptually Uniform Colour Maps
Peter Kovesi
Geophysics & Image Analysis Group
Centre for Exploration Targeting
School of Earth and Environment
The University of Western Australia

peter.kovesi@uwa.edu.au

and also on the paper arXiv:1509.03700 found here:

https://arxiv.org/abs/1509.03700
"""

import matplotlib

matplotlib.use('TkAgg')

import numpy as np

from collections import Iterable
from scipy import interpolate, optimize
from scipy.ndimage import filters
from matplotlib import pyplot, cm
from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap

from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import (delta_e_cie1976, delta_e_cie1994,
                                  delta_e_cie2000, delta_e_cmc)

def mpl_cmap(name, cielab_colors):
    to_rgb = lambda c: convert_color(LabColor(*c), sRGBColor)
    rgb_values = lambda c: (c.clamped_rgb_r, c.clamped_rgb_g, c.clamped_rgb_b)
    rgb_cmap = [to_rgb(c) for c in cielab_colors]
    rgb_cmap = [rgb_values(c) for c in rgb_cmap]
    return LinearSegmentedColormap.from_list(name, rgb_cmap)

def deltaE(spline, x0, x1):
    _deltaE = delta_e_cie1976
    return _deltaE(LabColor(*spline(x0)), LabColor(*spline(x1)))

class SplineND(object):
    def __init__(self, x, yy, kk=3, ss=0, *args, **kwargs):
        assert not hasattr(kwargs, 'k'), 'use kk'
        assert not hasattr(kwargs, 's'), 'use ss'
        if not isinstance(kk, Iterable):
            kk = [kk] * len(yy.T)
        if not isinstance(ss, Iterable):
            ss = [ss] * len(yy.T)
        ss = [s * len(x) for s in ss]
        self.splines = []
        for y, k, s in zip(yy.T, kk, ss):
            self.splines.append(interpolate.UnivariateSpline(
                x, y, k=k, s=s, *args, **kwargs))
    def __call__(self, x):
        return np.array([s(x) for s in self.splines]).T

def generate_cmap(name,
                  cielab_points,
                  npoints=256,
                  spline_orders=(2, 3, 3),
                  smoothing=(5, 10, 10),
                  plotinfo=False):

    # create smoothed spline based on input cielab points
    x = np.linspace(0, 1, len(cielab_points))
    spline = SplineND(x, cielab_points, spline_orders, smoothing)

    # calculate deltaE for each step, force dE to be non-zero
    xx = np.linspace(0, 1, npoints)
    dE = np.array([deltaE(spline, x0, x1) for x0, x1 in zip(xx[:-1], xx[1:])])
    dE[dE < 0.001] = 0.001

    # create accurate spline of cumulative sum of deltaE
    dEcsum = dE.cumsum()
    dEspline = interpolate.CubicSpline(dEcsum, xx[1:])

    # use spline to produce points along x that equally spaced in deltaE
    dEequicsum = np.linspace(dEcsum[0], dEcsum[-1], len(dEcsum))
    xxequi = np.concatenate([[xx[0]], dEspline(dEequicsum)])

    # use original spline to get cielab points equally spaced in E-space
    cielab_norm = spline(xxequi)

    # perform some extra smoothing of the resulting colors
    sigma = 0.5 * np.sqrt(len(cielab_norm))
    cielab_norm = filters.gaussian_filter1d(cielab_norm, sigma=sigma, axis=0)

    if plotinfo:
        fig, ax = pyplot.subplots(2,3)

        for i, l in enumerate(['lightness', 'green-red', 'blue-yellow']):
            ax[0,i].set_title(l)
            ax[0,i].plot(x, cielab_points[:,i], label='input', marker='o')
            ax[0,i].plot(xx, spline(xx)[:,i], label='spline', lw=4)
            ax[0,i].plot(xxequi, cielab_norm[:,i], label='equi')

        ax[1,0].set_xlabel('dE equi-distant')
        ax[1,0].set_ylabel('x')
        ax[1,0].plot(dEcsum, xx[1:], label='dEcsum', lw=4)
        ax[1,0].plot(dEequicsum, xxequi[1:])

    # return a matplotlib colormap
    return mpl_cmap(name, cielab_norm)


cmapdata = {
    #'KBGYW': {
    #    'cielab_points': np.array([
    #        (  0,   0,    0),
    #        ( 15,  49,  -64),
    #        ( 35,  67, -100),
    #        ( 45, -12,  -29),
    #        ( 60, -55,   60),
    #        ( 80, -20,   80),
    #        ( 90, -17,   40),
    #        (105,   0,    0)]),
    #    },
    #'KGB': {
    #    'cielab_points': np.array([
    #        ( 10,   0,    0),
    #        ( 50, -50,    0),
    #        ( 90,   0,  -50)]),
    #    'spline_orders': (2, 2, 2),
    #    },
    'macaw': {
        'rgb_points': np.array([
            (32, 32, 32),  # black
            (3, 156, 143),  # blue
            (76, 122, 9),  # green
            (225, 138, 2),  # orange
            (240, 164, 1),  # yellow
            (200, 202, 202),  # bluish-white
            (255, 255, 255),  # white
        ]),
        'spline_orders': (2, 3, 3),
        'smoothing': (50, 10, 10),
    },
    'macaw-isolum': {
        #'rgb_points': np.array([
        #    (32, 32, 32),  # black
        #    (3, 156, 143),  # blue
        #    (76, 122, 9),  # green
        #    (225, 138, 2),  # orange
        #    (240, 164, 1),  # yellow
        #    (200, 202, 202),  # bluish-white
        #    (255, 255, 255),  # white
        #]),
        #'cielab_points': np.array([
        #    [  1.22500263e+01,  -1.11877281e-04,  -2.08501245e-03],
        #    [  5.78254174e+01,  -3.71519150e+01,  -2.95909928e+00],
        #    [  4.62778614e+01,  -3.24822705e+01,   4.89168388e+01],
        #    [  6.51147179e+01,   2.55361420e+01,   7.04871038e+01],
        #    [  7.29456546e+01,   1.81239956e+01,   7.66668101e+01],
        #    [  8.11730475e+01,  -6.86684464e-01,  -2.48256768e-01],
        #    [  9.99999845e+01,  -4.59389408e-04,  -8.56145792e-03]]),
        'cielab_points': np.array([
            [ 80, -40,-30],
            [ 80, -32, 50],
            [ 80,  25, 70],
            [ 80,  18, 75],
            ]),
        'spline_orders': (1, 3, 3),
        'smoothing': (0, 70, 70),
    },
}

for name, data in cmapdata.items():
    if 'rgb_points' in data:
        cielab_points = []
        for color in cmapdata[name]['rgb_points']:
            lab = convert_color(sRGBColor(*[c/255 for c in color]), LabColor)
            cielab_points.append(lab.get_value_tuple())
        cmapdata[name]['cielab_points'] = np.array(cielab_points)
        del cmapdata[name]['rgb_points']
        print(name)
        print(data)

def cmap_comb(ax, cmap):
    nperiods, npoints, amplitude = 80, 50, 70
    x = np.linspace(0, nperiods * 2 * np.pi, npoints * nperiods)
    y = np.linspace(0, amplitude, npoints * 10)
    X, Y = np.meshgrid(x, y)
    img = X +  Y * np.sin(X) * (Y**2 / Y.max()**2)
    ax.imshow(img, cmap=cmap, aspect=2, origin='lower', vmin=x[0], vmax=x[-1])
    ax.set_title(cmap.name)
    ax.set_axis_off()

fig, ax = pyplot.subplots(len(cmapdata), figsize=(12,4*len(cmapdata)))

for i, (name, data) in enumerate(cmapdata.items()):
    cmap = generate_cmap(name, plotinfo=True, **data)
    cmap_comb(ax[i], cmap)

pyplot.show()
