import matplotlib
matplotlib.use('wxAgg')

import numpy as np

from cielab_colormap import ColorMap
from matplotlib import pyplot
from matplotlib.colors import LinearSegmentedColormap


def cmap_comb(ax, cmap):
    nperiods, npoints, amplitude = 80, 50, 70
    x = np.linspace(0, nperiods * 2 * np.pi, npoints * nperiods)
    y = np.linspace(0, amplitude, npoints * 10)
    X, Y = np.meshgrid(x, y)
    img = X +  Y * np.sin(X) * (Y**2 / Y.max()**2)
    ax.imshow(img, cmap=cmap, aspect='auto', origin='lower', vmin=x[0], vmax=x[-1])
    ax.set_title(cmap.name)
    ax.set_axis_off()

cielab_points = np.array(
    [[ 12,   0,   0],
     [ 57, -37,  -2],
     [ 46, -32,  48],
     [ 65,  25,  70],
     [ 72,  18,  76],
     [ 81,   0,   0],
     [ 99,   0,   0]]).transpose()

x = np.linspace(0, 1, len(cielab_points[0]))

cmap = ColorMap(x, cielab_points)

'''
print(cmap.rgb())
for imdata in cmap.imdata():
    print(imdata)
'''

fig, axs = pyplot.subplots(5)
for ax in axs:
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

for ax, data in zip(axs, cmap.imdata()):
    pt = ax.imshow(data, origin='lower', aspect='auto', zorder=-1,
                   interpolation='gaussian')

macaw = LinearSegmentedColormap.from_list('macaw', cmap.rgb())
cmap_comb(axs[-2], macaw)

cmap.delta_e = None
cmap.update_corrected_colors()
macaw = LinearSegmentedColormap.from_list('macaw', cmap.rgb())
cmap_comb(axs[-1], macaw)

pyplot.show()
