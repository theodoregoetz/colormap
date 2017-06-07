import matplotlib
matplotlib.use('wxAgg')

import numpy as np

from cielab_colormap import ColorMap


cielab_points = np.array([
    [  1.22500263e+01,  -1.11877281e-04,  -2.08501245e-03],
    [  5.78254174e+01,  -3.71519150e+01,  -2.95909928e+00],
    [  4.62778614e+01,  -3.24822705e+01,   4.89168388e+01],
    [  6.51147179e+01,   2.55361420e+01,   7.04871038e+01],
    [  7.29456546e+01,   1.81239956e+01,   7.66668101e+01],
    [  8.11730475e+01,  -6.86684464e-01,  -2.48256768e-01],
    [  9.99999845e+01,  -4.59389408e-04,  -8.56145792e-03]]).transpose()

x = np.linspace(0, 1, len(cielab_points[0]))

cmap = ColorMap(x, cielab_points)

print(cmap.rgb())
for imdata in cmap.imdata():
    print(imdata)


self.plots[l].axes.imshow(imdata, extent=ext,
                origin='lower', aspect='auto', zorder=-1,
                interpolation='gaussian')
