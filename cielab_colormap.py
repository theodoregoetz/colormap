import numpy as np

from scipy import interpolate
from skimage import color


class ColorMap(object):
    CHANNELS = ['lightness', 'red-green', 'blue-yellow']

    def __init__(self, x, cielab_colors, k=(3, 3, 3), s=(0, 0, 0)):
        self.splines = []
        for values, kk, ss in zip(cielab_colors, k, s):
            self.splines.append(self._spline(x, values, kk, ss))

    def _spline(self, x, y, k, s):
        return interpolate.Univariate(x, y, k=k, s=s)

    def cielab(self, npoints=256):
        x = np.linspace(0, 1, npoints)
        return np.stack([s(x) for s in self.splines], axis=1)

    def delta_e_correction(self, cielab_colors, delta_e=color.delta_e_cie1994):

        if delta_e is not None:
            # calculate dE for each step, force dE to be non-zero
            de = delta_e(cielab_colors[:-1], cielab_colors[1:])
            de[de < 0.001] = 0.001

            de_csum = de.cumsum()
            de_spline = interpolate.CubicSpline(de_csum, x[1:])
            de_equicsum = np.linspace(de_csum[0], de_csum[-1], len(de_csum))
            x_equi = np.concatenate([xx[:1], de_spline(de_equicumsum))

            for i, s in enumerate(self.splines):
                cielab_colors[i, ...] = s(x_equi)

        return cielab_colors

    def rgb(self, npoints=256, delta_e=color.delta_e_cie1994):
        return color.lab2rgb(self.cielab(npoints, delta_e))

    def imdata(self, dim=(40, 64)):




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


class ColorMapCIELab(object):
    def __init__(self, npoints=40):
        xdim, ydim = npoints, 64
        self.x = np.linspace(0, 1, xdim)
        self.ll = np.linspace(   0, 100, ydim).reshape((-1,1))
        self.aa = np.linspace(-128, 128, ydim).reshape((-1,1))
        self.bb = np.linspace(-128, 128, ydim).reshape((-1,1))
        self._imdata = np.empty((ydim, xdim, 3))
        self._alpha = np.empty((ydim, xdim))

    def alpha(self, data, alpha_at_limits=1.0):
        self._alpha[...] = 1
        for channel in np.rollaxis(data, axis=-1):
            self._alpha[channel<0.001] = alpha_at_limits
            self._alpha[channel>0.999] = alpha_at_limits
        return self._alpha

    def cielab_colors(self, *splines, x=None):
        return np.array([s(x or self.x) for s in splines])

    def imdata(self, cielab_colors):
        l, a, b = cielab_colors
        extent = [self.x.min(), self.x.max(), self.ll.min(), self.ll.max()]
        self._imdata[:,:,0] = self.ll
        self._imdata[:,:,1] = a
        self._imdata[:,:,2] = b
        data = color.lab2rgb(self._imdata)
        ret = [(np.dstack([data, self.alpha(data)]), copy(extent))]

        extent[-2] = self.aa.min()
        extent[-1] = self.aa.max()
        self._imdata[:,:,0] = l
        self._imdata[:,:,1] = self.aa
        data = color.lab2rgb(self._imdata)
        ret += [(np.dstack([data, self.alpha(data)]), copy(extent))]

        extent[-2] = self.bb.min()
        extent[-1] = self.bb.max()
        self._imdata[:,:,1] = a
        self._imdata[:,:,2] = self.bb
        data = color.lab2rgb(self._imdata)
        ret += [(np.dstack([data, self.alpha(data)]), copy(extent))]

        return ret

    def mpl_cmap(self, name, *splines, npoints=256):
        x = np.linspace(0, 1, npoints)
        cielab_colors = self.cielab_colors(*splines, x=x)
        rgb = color.lab2rgb(cielab_colors.reshape(-1,1,3))
        return LinearSegmentedColormap.from_list(name, rgb.squeeze())
