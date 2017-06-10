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

class LinePlot(wx.Panel):
    def __init__(self, parent, name, xmin, xmax, ymin, ymax, npoints=5):
        super().__init__(parent)

        self.parent = parent
        self.name = name

        xdpi, ydpi = wx.ScreenDC().GetPPI()
        self.figure = Figure(figsize=(1, 1), dpi=xdpi, tight_layout=True,
                             subplotpars=SubplotParams(left=0.01, right=0.99,
                                                       bottom=0.01, top=0.99))
        self.axes = self.figure.add_subplot(1,1,1)
        self.axes.set_ylabel(self.name)

        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.x = np.linspace(self.xmin, self.xmax, npoints)
        self.y = rand.uniform(self.ymin, self.ymax, len(self.x))

        self.axes.set_xlim(self.xmin, self.xmax)
        self.axes.set_ylim(self.ymin, self.ymax)
        self.axes.xaxis.set_visible(False)
        self.axes.yaxis.set_ticks([])
        self.axes.yaxis.set_ticklabels([])
        self.axes.autoscale(False)

        self.plt, = self.axes.plot(self.x, self.y, marker='o')

        self.canvas = Canvas(self, wx.ID_ANY, self.figure)

        self.layout()
        self.connect()

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
                self.parent.save_cache(self.name)
                self._cache = {'x': self.x.copy(), 'y': self.y.copy()}
                self._index = self.pick(event.x, event.y, event.xdata, event.ydata)
                self._dragging = True
                if self._index is None:
                    self.add(event.xdata, event.ydata)
                else:
                    self.on_motion_notify(event)
            elif event.button == 3:
                idx = self.pick(event.x, event.y, event.xdata, event.ydata)
                if idx is not None:
                    self.delete(idx)

    def add(self, x, y):
        if self._dragging:
            self._index = 0
        self.x = np.concatenate([[x], self.x])
        self.y = np.concatenate([[y], self.y])
        self.update()
        self.parent.add(self.name, x)

    def delete(self, idx):
        if idx == 0:
            if self.x[idx] == self.x[idx + 1]:
                self.x = np.delete(self.x, idx)
                self.y = np.delete(self.y, idx)
                self.update()
                self.parent.delete(self.name, idx)
        elif idx == len(self.x) - 1:
            if self.x[idx] == self.x[idx - 1]:
                self.x = np.delete(self.x, idx)
                self.y = np.delete(self.y, idx)
                self.update()
                self.parent.delete(self.name, idx)
        else:
            self.x = np.delete(self.x, idx)
            self.y = np.delete(self.y, idx)
            self.update()
            self.parent.delete(self.name, idx)

    def on_motion_notify(self, event):
        if self._dragging:
            if self._index is not None:
                if event.inaxes:
                    x, y = event.xdata, event.ydata
                else:
                    transDataInv = self.axes.transData.inverted()
                    x, y = transDataInv.transform((event.x, event.y))
                    x = min(max(x, self.xmin), self.xmax)
                    y = min(max(y, self.ymin), self.ymax)

                if self._index not in [0, len(self.x) - 1]:
                    self.x[self._index] = x
                self.y[self._index] = y

                self.update()
                if self._index not in [0, len(self.x) - 1]:
                    self.parent.move(self.name, self._index, x)
                else:
                    self.parent.update_colormap_data()
                    self.parent.update_colormaps()

    def on_button_release(self, event):
        if self._dragging:
            self.on_motion_notify(event)
            self._dragging = False
            self._index = None

    def on_figure_leave(self, event):
        if self._dragging:
            self._dragging = False
            self._index = None
            self.x = self._cache['x'].copy()
            self.y = self._cache['y'].copy()
            self.update()
            del self._cache
            self.parent.restore_cache(self.name)

    def pick(self, x, y, xdata, ydata):
        idx = ((self.x - xdata)**2 + (self.y - ydata)**2).argmin()
        xpx, ypx = self.axes.transData.transform([self.x[idx], self.y[idx]])
        distsq = (xpx - x)**2 + (ypx - y)**2
        if distsq < 100:
            return idx

    def update_points(self):
        if any(self.x[:-1] > self.x[1:]):
            order = np.argsort(self.x)
            if self._dragging:
                self._index = np.argwhere(order == self._index).flat[0]
            self.x = self.x[order]
            self.y = self.y[order]
        if self.x[0] != self.xmin:
            self.x = np.concatenate([[self.xmin], self.x])
            self.y = np.concatenate([self._cache['y'][:1], self.y])
            if self._dragging:
                if self._index is not None:
                    self._index += 1
        if self.x[-1] != self.xmax:
            self.x = np.concatenate([self.x, [self.xmax]])
            self.y = np.concatenate([self.y, self._cache['y'][-1:]])
        self.plt.set_data(self.x, self.y)

    def update(self):
        self.update_points()
        self.canvas.draw()

    def OnPaint(self, event):
        self.canvas.draw()
        event.Skip()


class SplinePlot(LinePlot):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.spline_x = np.linspace(self.xmin, self.xmax, 200)
        self.spline_plt, = self.axes.plot(self.spline_x,
                                          np.zeros(self.spline_x.shape))
        self.update_spline()

    def update_spline(self):
        k = min(len(self.x)-1, 3)
        self.spline = interpolate.UnivariateSpline(self.x, self.y, k=k, s=100)
        self.spline_plt.set_data(self.spline_x, self.spline(self.spline_x))

    def update(self):
        self.update_points()
        self.update_spline()
        self.canvas.draw()


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


class ColorMapControlCIELab(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.plots = {
            'lightness': SplinePlot(self, 'lightness', 0, 1, 0, 100),
            'green-red': SplinePlot(self, 'green-red', 0, 1, -128, 128),
            'blue-yellow': SplinePlot(self, 'blue-yellow', 0, 1, -128, 128)}
        self.init_colormaps()
        self.layout()

    def layout(self):
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.plots['lightness'], 1, wx.ALL | wx.EXPAND)
        vbox.Add(self.plots['green-red'], 1, wx.ALL | wx.EXPAND)
        vbox.Add(self.plots['blue-yellow'], 1, wx.ALL | wx.EXPAND)
        self.SetSizer(vbox)
        self.Layout()

    def save_cache(self, name):
        for plot in (p for n, p in self.plots.items() if n != name):
                plot._cache = {'x': plot.x.copy(), 'y': plot.y.copy()}

    def add(self, name, x):
        for plot in (p for n, p in self.plots.items() if n != name):
            plot.x = np.concatenate([[x], plot.x])
            plot.y = np.concatenate([[0.5 * (plot.ymin + plot.ymax)], plot.y])
            plot.update()
        self.update_colormaps()

    def delete(self, name, idx):
        for plot in (p for n, p in self.plots.items() if n != name):
            plot.x = np.delete(plot.x, idx)
            plot.y = np.delete(plot.y, idx)
            plot.update()
        self.update_colormaps()

    def move(self, name, idx, x):
        for plot in (p for n, p in self.plots.items() if n != name):
            plot.x[idx] = x
            plot.update()
        self.update_colormaps()

    def restore_cache(self, name):
        for plot in (p for n, p in self.plots.items() if n != name):
            plot.x = plot._cache['x'].copy()
            plot.y = plot._cache['y'].copy()
            plot.update()
            del plot._cache
        self.update_colormaps()

    def init_colormaps(self):
        channels = ['lightness', 'green-red', 'blue-yellow']
        self.cmap = ColorMapCIELab()
        self.update_colormap_data()
        self.field_plots = {}
        for l, (imdata, ext) in zip(channels, self.imdata):
            self.field_plots[l] = self.plots[l].axes.imshow(imdata, extent=ext,
                origin='lower', aspect='auto', zorder=-1,
                interpolation='gaussian')

    def update_colormap_data(self):
        channels = ['lightness', 'green-red', 'blue-yellow']
        splines = [self.plots[l].spline for l in channels]
        cielab_colors = self.cmap.cielab_colors(*splines)
        self.imdata = self.cmap.imdata(cielab_colors)

    def update_colormaps(self):
        channels = ['lightness', 'green-red', 'blue-yellow']
        self.update_colormap_data()
        for l, (imdata, ext) in zip(channels, self.imdata):
            self.field_plots[l].set_array(imdata)
            self.field_plots[l].axes.figure.canvas.draw()


class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, wx.ID_ANY, 'Main Window', size=(500, 600))
        self.cmapctl = ColorMapControlCIELab(self)
        self.layout()

    def layout(self):
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.cmapctl, 1, wx.ALL | wx.EXPAND)
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
