from util.tensorflow_summary import extract_tensorflow_summary
import tensorflow as tf
from options import get_options
from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np

flags = get_options('plot')

class Application:
    def __init__(self):
        pass

    def run(self):
        r = extract_tensorflow_summary(flags.log_dir, flags.metrics)
        plt.plot(np.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=10))
        plt.show()
        pass

if __name__ == '__main__':
    Application().run()
