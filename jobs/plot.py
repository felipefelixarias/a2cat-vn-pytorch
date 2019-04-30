#!/usr/bin/python3

import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy import interpolate

def load_metrics(file):
    import csv

    metrics = defaultdict(lambda: ([], []))
    for line in csv.reader(file):
        name = line[0]
        count = int(line[1])
        times = list(map(int, line[2:(2 + count)]))
        values = list(map(float, line[(2 + count):]))
        metrics[name][0].extend(times)
        metrics[name][1].extend(values)
    return metrics

def interpolate_data(data, factor = None):
    x, y = data
    #tck = interpolate.splrep(x, y, s=0)    
    spl = interpolate.UnivariateSpline(*data, s = factor)
    xmin, xmax = min(x), max(x)
    xnew = np.arange(xmin, xmax, (xmax-xmin) / 500)
    return xnew, spl(xnew)

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

if __name__ == "__main__":
    import sys

    with open('/models/%s/metrics.txt' % sys.argv[1], 'r') as f:
        metrics = load_metrics(f)

    def plot_single_metric(data, color, label, smoothing_factor = None):
        #plt.plot(*data, c = color)
        plt.plot(*data, c = color, linewidth = 0.3)
        plt.plot(*interpolate_data(data, smoothing_factor), c = color, label = label)
        plt.axhline(y=max(interpolate_data(data)[1]), color = lighten_color(color, 1.2), linewidth = 0.3)
        
    fig = plt.figure()
    data, color = (metrics['reward'], 'r')
    plot_single_metric(data, color,  'a2c-unreal', 800)
    plt.xlabel('timestep')
    plt.ylabel('reward')
    plt.grid(linestyle='--')
    #plt.xlim(-, 1.2e7)
    plt.legend()
    plt.show()