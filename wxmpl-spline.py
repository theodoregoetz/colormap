import matplotlib as mpl
mpl.use('wxAgg')

import numpy as np
import warnings
import wx

from copy import copy
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as Canvas
from matplotlib.colors import LinearSegmentedColormap
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

    def add(self, x, y=None):
        self.x = np.concatenate([[self._clip_x(x)], self.x])
        if y is None:
            y = 0.5 * (self.ymin + self.ymax)
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
                 delta_e=color.deltaE_ciede94,
                 smoothing=0,
                 npoints=256,
                 imdim=(40, 64)):
        self.x = x
        self.colors = np.asarray(colors).reshape(3, -1)
        self.delta_e = delta_e
        self.smoothing = smoothing
        self.npoints = npoints
        self.imdim = imdim

        self.splines = [
            BoundedSpline(npoints, self.x, self.colors[0], (0, 1, 0, 100)),
            BoundedSpline(npoints, self.x, self.colors[1], (0, 1, -128, 128)),
            BoundedSpline(npoints, self.x, self.colors[2], (0, 1, -128, 128))]

        ncorrpoints = 100
        self._xx = np.linspace(0, 1, ncorrpoints)
        self._colors_uncorr = np.empty((ncorrpoints, 3))
        for i in (0, 1, 2):
            self._update_uncorrected_colors(i)

        self._update_x_spline()

        self._cmap_x = np.linspace(0, 1, self.npoints)
        self._cmap_colors = np.empty((npoints, 1, 3))

        self._ll = np.linspace(   0, 100, self.imdim[1]).reshape((-1,1))
        self._aa = np.linspace(-128, 128, self.imdim[1]).reshape((-1,1))
        self._bb = np.linspace(-128, 128, self.imdim[1]).reshape((-1,1))
        self._imx = np.linspace(0, 1, self.imdim[0])
        self._labdata = np.empty((3, self.imdim[1], self.imdim[0], 3))
        self._rgbdata = np.empty((3, self.imdim[1], self.imdim[0], 3), dtype=np.uint8)

        self._labdata[0, :, :, 0] = self._ll
        self._labdata[1, :, :, 1] = self._aa
        self._labdata[2, :, :, 2] = self._bb

    def _update_uncorrected_colors(self, i):
        self._colors_uncorr[..., i] = self.splines[i](self._xx)

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

    def update(self):
        for i in (0,1,2):
            self._update_uncorrected_colors(i)
        self._update_x_spline()

    def rgb(self):
        for i, s in enumerate(self.splines):
            self._cmap_colors[..., 0, i] = s(self._x_spline(self._cmap_x))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return color.lab2rgb(self._cmap_colors).squeeze()

    def imdata(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = self._imx
            l = self.splines[0](x)
            a = self.splines[1](x)
            b = self.splines[2](x)

            self._labdata[0, :, :, 1] = a
            self._labdata[0, :, :, 2] = b
            self._rgbdata[0, :, :, :] = (color.lab2rgb(self._labdata[0]) * 0xff).astype(np.uint8)

            self._labdata[1, :, :, 0] = l
            self._labdata[1, :, :, 2] = b
            self._rgbdata[1, :, :, :] = (color.lab2rgb(self._labdata[1]) * 0xff).astype(np.uint8)

            self._labdata[2, :, :, 0] = l
            self._labdata[2, :, :, 1] = a
            self._rgbdata[2, :, :, :] = (color.lab2rgb(self._labdata[2]) * 0xff).astype(np.uint8)

            return self._rgbdata


class SplinePlot(wx.Panel):
    def __init__(self, parent, spline, xlabel=None, ylabel=None):
        super().__init__(parent)

        self.parent = parent
        self.spline = spline
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

        tol = 0.02
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

    def add(self, x, y=None):
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
                ret = self._index, x, y
                self._index = self.spline.move(self._index, x, y)
                self.update()
                return ret

    def on_button_release(self, event):
        if self._dragging:
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
        if distsq < 40**2:
            return ipick

    def update(self):
        self.plt.set_data(self.spline.x, self.spline.y)
        self.spline_plt.set_data(self.spline.spline_x, self.spline.spline_y)


class ColorMapSplinePlot(SplinePlot):
    def on_button_press(self, event):
        if event.inaxes:
            if event.button == 1:
                for p in self.parent.plots:
                    if self.GetId() != p.GetId():
                        p.spline.save_cache()
                ret = super().on_button_press(event)
            if event.button == 3:
                ret = super().on_button_press(event)
                self.parent.update()

    def add(self, x, y):
        super().add(x, y)
        for p in self.parent.plots:
            if self.GetId() != p.GetId():
                SplinePlot.add(p, x)
        self.parent.update()

    def delete(self, idx):
        super().delete(idx)
        for p in self.parent.plots:
            if self.GetId() != p.GetId():
                SplinePlot.delete(p, idx)
        self.parent.update()

    def on_motion_notify(self, event):
        res = super().on_motion_notify(event)
        if res is not None:
            idx, x, y = res
            for p in self.parent.plots:
                if self.GetId() != p.GetId():
                    p.spline.move(idx, x)
                    p.update()
            self.parent.update()

    def on_figure_leave(self, event):
        if self._dragging:
            for p in self.parent.plots:
                if self.GetId() != p.GetId():
                    if p.spline.load_cache():
                        p.update()
            super().on_figure_leave(event)
            self.parent.update()


class DECorrPlot(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.init_canvas()
        self.layout()

    def init_canvas(self):
        xdpi, ydpi = wx.ScreenDC().GetPPI()
        self.figure = Figure(figsize=(1, 1), dpi=xdpi, tight_layout=True,
                             subplotpars=SubplotParams(left=0.01, right=0.99,
                                                       bottom=0.01, top=0.99))
        self.axes = self.figure.add_subplot(1,1,1)

        self.axes.set_xlabel(r'uncorrected')
        self.axes.set_ylabel(r'corrected')

        self.axes.set_xlim(0, 1)
        self.axes.set_ylim(-0.1, 1.1)
        self.axes.autoscale(False)
        self.axes.xaxis.set_ticks([0,0.5,1])
        self.axes.yaxis.set_ticks([0,0.5,1])

        self._x = np.linspace(0, 1, 300)
        self.plot, = self.axes.plot((0,1), (0,1), color='black')

        self.canvas = Canvas(self, wx.ID_ANY, self.figure)
        self.update(lambda x: x)

    def layout(self):
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.canvas, 1, wx.ALL | wx.EXPAND, 0)
        self.SetSizerAndFit(vbox)
        self.Layout()

    def update(self, spline):
        self.plot.set_data(self._x, spline(self._x))
        self.canvas.draw()


class CombPlot(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.init_canvas()
        self.layout()

    def init_canvas(self):
        xdpi, ydpi = wx.ScreenDC().GetPPI()
        self.figure = Figure(figsize=(1, 1), dpi=xdpi, tight_layout=True,
                             subplotpars=SubplotParams(left=0.01, right=0.99,
                                                       bottom=0.01, top=0.99))
        self.axes = self.figure.add_subplot(1,1,1)

        nperiods, npoints, amplitude = 80, 50, 70
        x = np.linspace(0, nperiods * 2 * np.pi, npoints * nperiods)
        y = np.linspace(0, amplitude, npoints * 10)
        X, Y = np.meshgrid(x, y)
        img = X +  Y * np.sin(X) * (Y**2 / Y.max()**2)
        self.plot = self.axes.imshow(img, origin='lower', aspect='auto',
                                     vmin=x[0], vmax=x[-1])
        self.axes.set_axis_off()

        self.canvas = Canvas(self, wx.ID_ANY, self.figure)

    def layout(self):
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.canvas, 1, wx.ALL | wx.EXPAND, 0)
        self.SetSizerAndFit(vbox)
        self.Layout()

    def update(self, cmap):
        self.plot.set_cmap(LinearSegmentedColormap.from_list('_', cmap.rgb()))
        self.canvas.draw()


class ColorMapControlPlots(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        # Macaw
        colors = np.array(
            [[ 12,   0,   0],
             [ 57, -37,  -2],
             [ 46, -32,  48],
             [ 65,  25,  70],
             [ 72,  18,  76],
             [ 81,   0,   0],
             [ 99,   0,   0]]).transpose()
        # Test
        #colors = np.array(
        #    [[ 50,   0,   0],
        #     [ 50,   0,   0],
        #     [ 50,   0,   0],
        #     [ 50,   0,   0],
        #     [  0,   0,   0],
        #     [ 50,   0,   0],
        #     [ 50,   0,   0],
        #     [ 50,   0,   0],
        #     [ 50,   0,   0],
        #     [100,   0,   0],
        #     [ 50,   0,   0],
        #     [ 50,   0,   0],
        #     [ 50,   0,   0],
        #     [ 50,   0,   0]]).transpose()
        ## Test
        #colors = np.array(
        #    [[  0,  50,   0],
        #     [ 50,  50,   0],
        #     [100,   0,  50],
        #     [ 50, -50,  50],
        #     [  0, -50,  50]]).transpose()
        x = np.linspace(0, 1, len(colors[0]))
        self.cmap = ColorMap(x, colors)

        self.plots = [
            ColorMapSplinePlot(self, self.cmap.splines[0], ylabel='lightness'),
            ColorMapSplinePlot(self, self.cmap.splines[1], ylabel='green-red'),
            ColorMapSplinePlot(self, self.cmap.splines[2], ylabel='blue-yellow')]

        self.implots = []
        for imdata, plot in zip(self.cmap.imdata(), self.plots):
            extent = list(plot.axes.get_xlim()) + list(plot.axes.get_ylim())
            self.implots.append(plot.axes.imshow(imdata, extent=extent,
                                                 origin='lower', aspect='auto',
                                                 zorder=-1,
                                                 interpolation='gaussian'))

        self.de_corr_plot = DECorrPlot(self)
        self.de_corr_plot.update(self.cmap._x_spline)

        self.comb_plot = CombPlot(self)
        self.comb_plot.update(self.cmap)

        self.layout()

    def layout(self):

        vbox0 = wx.BoxSizer(wx.VERTICAL)

        hbox0 = wx.BoxSizer(wx.HORIZONTAL)

        vbox1 = wx.BoxSizer(wx.VERTICAL)
        vbox1.Add(self.plots[0], 1, wx.ALL | wx.EXPAND)
        vbox1.Add(self.plots[1], 1, wx.ALL | wx.EXPAND)
        vbox1.Add(self.plots[2], 1, wx.ALL | wx.EXPAND)
        hbox0.Add(vbox1, 2, wx.ALL | wx.EXPAND)

        vbox2 = wx.BoxSizer(wx.VERTICAL)
        vbox2.Add(self.de_corr_plot, 1, wx.ALL | wx.EXPAND)
        vbox2.Add(wx.Panel(self), 2, wx.ALL | wx.EXPAND)
        hbox0.Add(vbox2, 1, wx.ALL | wx.EXPAND)

        vbox0.Add(hbox0, 3, wx.ALL | wx.EXPAND)
        vbox0.Add(self.comb_plot, 1, wx.ALL | wx.EXPAND)

        self.SetSizerAndFit(vbox0)
        self.Layout()

    def update(self):
        for im, data in zip(self.implots, self.cmap.imdata()):
            im.set_array(data)
            im.figure.canvas.draw()
        self.cmap.update()
        self.de_corr_plot.update(self.cmap._x_spline)
        self.comb_plot.update(self.cmap)


class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, wx.ID_ANY, 'Main Window', size=(750, 600))
        self.cmapctl = ColorMapControlPlots(self)
        self.layout()

        self.cmapctl.update()

    def layout(self):
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.cmapctl, 1, wx.ALL | wx.EXPAND)
        self.SetSizer(hbox)
        self.Layout()


class Application(wx.App):
    def OnInit(self):
        mainFrame = MainFrame()
        mainFrame.Show(True)
        return True


if __name__ == '__main__':
    app = Application(False)
    app.MainLoop()
