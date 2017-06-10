import numpy as np
import warnings

from scipy import interpolate
from skimage import color


class ColorMap(object):
    def __init__(self, x, colors, k=(5, 5, 5), s=(0, 0, 0),
                 delta_e=color.deltaE_ciede94, smoothing=50, npoints=256,
                 imdim=(40, 64)):
        self.x = x
        self.colors = colors
        self.k = k
        self.s = s
        self.delta_e = delta_e
        self.smoothing = smoothing
        self.npoints = npoints
        self.imdim = imdim

        self._splines = [None] * 3
        for i in (0, 1, 2):
            self._update_spline(i)

        ncorrpoints = 100
        self._xx = np.linspace(0, 1, ncorrpoints)
        self._colors_uncorr = np.empty((ncorrpoints, 3))
        self._colors_corr = np.empty((ncorrpoints, 3))
        for i in (0, 1, 2):
            self._update_uncorrected_colors(i)

        self._update_x_spline()
        self._update_corrected_colors()

        self._cmap_x = np.linspace(0, 1, self.npoints)
        self._cmap_colors = np.empty((npoints, 1, 3))

        self._ll = np.linspace(   0, 100, self.imdim[1]).reshape((-1,1))
        self._aa = np.linspace(-128, 128, self.imdim[1]).reshape((-1,1))
        self._bb = np.linspace(-128, 128, self.imdim[1]).reshape((-1,1))
        self._imx = np.linspace(0, 1, self.imdim[0])
        self._imdata = np.empty((self.imdim[1], self.imdim[0], 3))

    def _update_spline(self, i):
        self._splines[i] = interpolate.UnivariateSpline(
            self.x, self.colors[i],
            k=min(len(self.x) - 1, self.k[i]), s=self.s[i])

    def _update_uncorrected_colors(self, i):
        self._colors_uncorr[..., i] = self._splines[i](self._xx)

    def _update_x_spline(self):
        if self.delta_e is None:
            self._x_spline = lambda x: x
        else:
            de = self.delta_e(self._colors_uncorr[:-1], self._colors_uncorr[1:])
            de[de < 0.001] = 0.001
            de_csum = de.cumsum() - de[0]
            de_csum /= de_csum[-1]
            self._x_spline = interpolate.UnivariateSpline(de_csum, self._xx[1:],
                                                          s=self.smoothing)

    def _update_corrected_colors(self):
        for i, s in enumerate(self._splines):
            self._colors_corr[..., i] = s(self._x_spline(self._xx))

    def update_corrected_colors(self):
        self._update_x_spline()
        self._update_corrected_colors()

    def update_channel(self, i):
        self._update_spline(i)
        self._update_uncorrected_colors(i)
        self.update_corrected_colors()

    def rgb(self):
        for i, s in enumerate(self._splines):
            self._cmap_colors[..., 0, i] = s(self._x_spline(self._cmap_x))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return color.lab2rgb(self._cmap_colors).squeeze()

    def imdata(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self._imdata[:, :, 0] = self._ll
            self._imdata[:, :, 1] = self._splines[1](self._x_spline(self._imx))
            self._imdata[:, :, 2] = self._splines[2](self._x_spline(self._imx))
            yield color.lab2rgb(self._imdata)

            self._imdata[:, :, 0] = self._splines[0](self._x_spline(self._imx))
            self._imdata[:, :, 1] = self._aa
            yield color.lab2rgb(self._imdata)

            self._imdata[:, :, 1] = self._splines[1](self._x_spline(self._imx))
            self._imdata[:, :, 2] = self._bb
            yield color.lab2rgb(self._imdata)
