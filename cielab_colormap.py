import numpy as np
import warnings

from enum import Enum
from scipy import interpolate
from skimage import color


class ColorMap(object):
    def __init__(self, x, colors, k=(3, 3, 3), s=(0, 0, 0), npoints=10, #256,
                 delta_e=color.deltaE_ciede94, imdim=(40, 64)):
        self.x = x
        self.colors = colors
        self.k = k
        self.s = s
        self.npoints = npoints
        self.delta_e = delta_e
        self.imdim = imdim

        self._splines = [None] * 3
        for i in (0, 1, 2):
            self._update_spline(i)

        self._xx = np.linspace(0, 1, self.npoints)
        self._colors_uncorr = np.empty((self.npoints, 3))
        self._colors_corr = np.empty((self.npoints, 3))
        for i in (0, 1, 2):
            self._update_colors(i)

        self._ll = np.linspace(   0, 100, self.imdim[1]).reshape((-1,1))
        self._aa = np.linspace(-128, 128, self.imdim[1]).reshape((-1,1))
        self._bb = np.linspace(-128, 128, self.imdim[1]).reshape((-1,1))
        self._imdata = np.empty((self.imdim[1], self.imdim[0], 3))

    def _update_spline(self, i):
        self._splines[i] = interpolate.UnivariateSpline(
            self.x, self.colors[i], k=min(len(self.x) - 1, self.k[i]),
            s=self.s[i])

    def _update_colors(self, i):
        self._colors_uncorr[...,i] = self._splines[i](self._xx)
        #self._update_delta_e_correction()

    def _update_delta_e_correction(self):
        de = self.delta_e(self._colors_uncorr.T[:-1,:], self._colors_uncorr.T[1:,:])
        de[de < 0.001] = 0.001
        de_csum = de.cumsum()
        de_spline = interpolate.CubicSpline(de_csum, self._xx[1:])
        de_equicsum = np.linspace(de_csum[0], de_csum[-1], len(de_csum))
        x_equi = np.concatenate([self._xx[:1], de_spline(de_equicumsum)])
        for i, s in enumerate(self._splines):
            self._colors_corr[...,i] = s(x_equi)

    def update_channel(self, i):
        self._update_spline(i)
        self._update_colors(i)

    def rgb(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return color.lab2rgb(self._colors_uncorr.reshape(-1,1,3))

    def imdata(self):
        self._imdata[:,:,0] = self._ll
        self._imdata[:,:,1] = self._colors_corr[:,1]
        self._imdata[:,:,2] = self._colors_corr[:,2]
        yield color.lab2rgb(self._imdata)

        self._imdata[:,:,0] = self._colors_corr[:,0]
        self._imdata[:,:,1] = self._aa
        yield color.lab2rgb(self._imdata)

        self._imdata[:,:,1] = self._colors_corr[:,1]
        self._imdata[:,:,2] = self._bb
        yield color.lab2rgb(self._imdata)
