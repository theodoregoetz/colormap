import matplotlib as mpl
mpl.use('wxAgg')

import numpy as np
import wx

from copy import copy
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as Canvas
from matplotlib.figure import Figure, SubplotParams
from numpy import random as rand
from scipy import interpolate
from skimage import color


rand.seed(1)


class OrderedBoundedPoints(object):
    def __init__(self, x, y, extent):
        self.xmin, self.xmax = extent[:2]
        self.ymin, self.ymax = extent[2:]
        self.x = self._clip_x(np.asarray(x))
        self.y = self._clip_y(np.asarray(y))
        self._reorder()
        self._cache = None

    def save_cache(self):
        self._cache = (self.x.copy(), self.y.copy())

    def load_cache(self):
        if self._cache is None:
            return False
        else:
            self.x, self.y = self._cache
            self._cache = None
            return True

    def pick(self, x, y):
        i = ((self.x - x)**2 + (self.y - y)**2).argmin()
        return i, self.x[i], self.y[i]

    def move(self, i, x=None, y=None):
        if x is not None and i not in (0, len(self.x) - 1):
            self.x[i] = self._clip_x(x)
        if y is not None:
            self.y[i] = self._clip_y(y)
        self._redistribute()
        self._reorder()
        return self._moved_index(i)

    def add(self, x, y):
        self.x = np.concatenate([[self._clip_x(x)], self.x])
        self.y = np.concatenate([[self._clip_y(y)], self.y])
        self._redistribute()
        self._reorder()
        return self._moved_index(0)

    def delete(self, i):
        if i in (0, len(self.x) - 1):
            return False
        else:
            self.x = np.delete(self.x, i)
            self.y = np.delete(self.y, i)
            return True

    def _clip_x(self, x):
        return np.clip(x, self.xmin, self.xmax)

    def _clip_y(self, y):
        return np.clip(y, self.ymin, self.ymax)

    def _reorder(self):
        self._order = None
        if len(self.x) > 2:
            if any(self.x[:-1] > self.x[1:]):
                self._order = np.argsort(self.x)
                self.x = self.x[self._order]
                self.y = self.y[self._order]

    def _redistribute(self):
        if len(self.x) > 2:
            tol = 0.001
            dmin = 0.01 * (self.xmax - self.xmin)
            close = np.isclose(self.x[:-1], self.x[1:])
            if close[-1]:
                dx = tol * abs(self.x[-1] - self.x[-3])
                self.x[-2] = self.x[-1] - max(dx, dmin)
            close = np.argwhere(close[:-1]).flatten()
            if len(close):
                dx = tol * np.abs(self.x[close + 2] - self.x[close])
                self.x[close + 1] = self.x[close] + np.maximum(dx, dmin)

    def _moved_index(self, i):
        if self._order is None:
            return i
        else:
            inew = np.argwhere(self._order == i).flat
            return i if i in inew else inew[0]


class BoundedSpline(OrderedBoundedPoints):
    def __init__(self, spline_npoints, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spline_x = np.linspace(self.xmin, self.xmax, spline_npoints)
        self.update_spline()

    def update_spline(self):
        self.spline = interpolate.PchipInterpolator(self.x, self.y)
        #self.spline = interpolate.CubicSpline(self.x, self.y, bc_type='natural')

    @property
    def spline_y(self):
        return np.clip(self.spline(self.spline_x), self.ymin, self.ymax)

    def load_cache(self):
        ret = super().load_cache()
        if ret:
            self.update_spline()
        return ret

    def move(self, *args, **kwargs):
        ret = super().move(*args, **kwargs)
        self.update_spline()
        return ret

    def add(self, *args, **kwargs):
        ret = super().add(*args, **kwargs)
        self.update_spline()
        return ret

    def delete(self, *args, **kwargs):
        ret = super().delete(*args, **kwargs)
        if ret:
            self.update_spline()
        return ret

    def __call__(self, x):
        return self.spline(x)


class ColorMap(object):
    def __init__(self, x, colors,
                 delta_e=color.deltaE_ciede94, smoothing=50, npoints=256,
                 imdim=(40, 64)):
        self.x = x
        self.colors = np.asarray(colors).reshape(3, -1)
        self.delta_e = delta_e
        self.smoothing = smoothing
        self.npoints = npoints
        self.imdim = imdim

        self._splines = [
            BoundedSpline(npoints, self.x, self.colors[0], (0, 1, 0, 100)),
            BoundedSpline(npoints, self.x, self.colors[1], (0, 1, -128, 128)),
            BoundedSpline(npoints, self.x, self.colors[2], (0, 1, -128, 128))]

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


class SplinePlot(wx.Panel):
    def __init__(self, parent, x, y, extent, xlabel=None, ylabel=None,
                 spline_npoints=256):
        super().__init__(parent)

        self.parent = parent
        self.spline = BoundedSpline(spline_npoints, x, y, extent)

        self.init_canvas(xlabel, ylabel)
        self.layout()
        self.connect()

    def init_canvas(self, xlabel, ylabel):
        xdpi, ydpi = wx.ScreenDC().GetPPI()
        self.figure = Figure(figsize=(1, 1), dpi=xdpi, tight_layout=True,
                             subplotpars=SubplotParams(left=0.01, right=0.99,
                                                       bottom=0.01, top=0.99))
        self.axes = self.figure.add_subplot(1,1,1)

        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)

        tol = 0.005
        dx = tol * (self.spline.xmax - self.spline.xmin)
        dy = tol * (self.spline.ymax - self.spline.ymin)
        self.axes.set_xlim(self.spline.xmin - dx, self.spline.xmax + dx)
        self.axes.set_ylim(self.spline.ymin - dy, self.spline.ymax + dy)
        self.axes.autoscale(False)
        self.axes.xaxis.set_visible(False)
        self.axes.yaxis.set_ticks([])
        self.axes.yaxis.set_ticklabels([])

        self.plt, = self.axes.plot(self.spline.x, self.spline.y,
                                   linestyle='none', marker='o', color='black')


        self.spline_plt, = self.axes.plot(self.spline.spline_x,
                                          self.spline.spline_y, color='black')

        self.canvas = Canvas(self, wx.ID_ANY, self.figure)

    def layout(self):
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.canvas, 1, wx.ALL | wx.EXPAND, 0)
        self.SetSizerAndFit(vbox)

    def connect(self):
        self._dragging = False
        self._index = None
        self.canvas.mpl_connect('button_press_event',
                                lambda e: self.on_button_press(e))
        self.canvas.mpl_connect('motion_notify_event',
                                lambda e: self.on_motion_notify(e))
        self.canvas.mpl_connect('button_release_event',
                                lambda e: self.on_button_release(e))
        self.canvas.mpl_connect('figure_leave_event',
                                lambda e: self.on_figure_leave(e))

    def on_button_press(self, event):
        if event.inaxes:
            if event.button == 1:
                self.spline.save_cache()
                self._dragging = True
                self._index = self.pick(event)
                if self._index is None:
                    self.add(event.xdata, event.ydata)
                else:
                    self.on_motion_notify(event)
            elif event.button == 3:
                idx = self.pick(event)
                if idx is not None:
                    self.delete(idx)

    def add(self, x, y):
        i = self.spline.add(x, y)
        if self._dragging:
            self._index = i
        self.update()

    def delete(self, idx):
        if self.spline.delete(idx):
            self.update()

    def on_motion_notify(self, event):
        if self._dragging:
            if self._index is not None:
                if event.inaxes:
                    x, y = event.xdata, event.ydata
                else:
                    x, y = self.axes.transData.inverted().transform((event.x, event.y))
                self._index = self.spline.move(self._index, x, y)
                self.update()

    def on_button_release(self, event):
        if self._dragging:
            self.on_motion_notify(event)
            self._dragging = False
            self._index = None

    def on_figure_leave(self, event):
        if self._dragging:
            self._dragging = False
            self._index = None
            if self.spline.load_cache():
                self.update()

    def pick(self, event):
        ipick, xpick, ypick = self.spline.pick(event.xdata, event.ydata)
        xpx, ypx = self.axes.transData.transform((xpick, ypick))
        distsq = (xpx - event.x)**2 + (ypx - event.y)**2
        if distsq < 10**2:
            return ipick

    def update(self):
        self.plt.set_data(self.spline.x, self.spline.y)
        self.spline_plt.set_data(self.spline.spline_x, self.spline.spline_y)
        self.canvas.draw()


class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, wx.ID_ANY, 'Main Window', size=(500, 250))

        extent = [0, 255, 0, 255]
        x = [0,  50, 130, 185, 255]
        y = [0, 150,  33, 210, 255]

        self.plot = SplinePlot(self, x, y, extent, ylabel='spline')
        self.layout()

    def layout(self):
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.plot, 1, wx.ALL | wx.EXPAND)
        self.SetSizer(vbox)
        self.Layout()


class Application(wx.App):
    def OnInit(self):
        mainFrame = MainFrame()
        mainFrame.Show(True)
        return True


if __name__ == '__main__':
    app = Application(False)
    app.MainLoop()
