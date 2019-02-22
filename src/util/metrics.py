import tensorflow as tf
import os
from abc import abstractclassmethod
from collections import defaultdict

import matplotlib.pyplot as plt

class MetricHandlerBase:
    def __init__(self, name, *args, **kwargs):
        self.name = name

    @abstractclassmethod
    def collect(self, collection, time, mode = 'train'):
        pass

PLOT_METRICS = ['reward', 'episode_length']
class MatplotlibHandler(MetricHandlerBase):
    def __init__(self, *args, **kwargs):
        super().__init__('matplotlib', *args, **kwargs)
        self._metrics = defaultdict(lambda: ([], []))
        self._validation_metrics = defaultdict(lambda: ([], []))
        self._figures = dict()

    def collect(self, collection, time, mode = 'train'):
        store = self._metrics if mode == 'train' else self._validation_metrics
        for (tag, val) in collection:
            t, v = store[tag]
            t.append(time)
            v.append(val)
        self.plot()

    def _get_figure(self, name):
        if name in self._figures:
            fig = self._figures[name]
            plt.figure(num = fig.number, clear = True)
        else:
            fig = plt.figure()
            fig.canvas.set_window_title(name)
            self._figures[name] = fig
        return fig

    def plot(self):
        for name, metric in self._metrics.items():
            if name in PLOT_METRICS:
                self._get_figure(name)
                plt.plot(metric[0], metric[1], 'b')

                if name in self._validation_metrics:
                    metric = self._validation_metrics[name]
                    plt.plot(metric[0], metric[1], 'r')

class MetricWriter:
    class _MetricRecordFactory:
        def __init__(self, time, flush, mode):
            self._flush = flush
            self.collection = []
            self._time = time
            self.mode = mode

        def scalar(self, name, value):
            self.collection.append((name, value))
            return self

        def flush(self):
            self._flush(self.collection, self._time, self.mode)
            self.collection.clear()
            return self

    def __init__(self, use_tensorboard = True, logdir = './logs'):
        self._use_tensorboard = use_tensorboard
        self._logdir = logdir
        self.handlers = [MatplotlibHandler()]

        if logdir is not None and len(logdir) > 0:
            if not os.path.exists(logdir):
                os.mkdir(logdir)

        if use_tensorboard:
            assert logdir is not None and len(logdir) > 0
            self._tensorboard_writer = tf.summary.FileWriter(logdir)
    
    def record(self, time):
        return MetricWriter._MetricRecordFactory(time, self._flush, mode = 'train')

    def record_validation(self, time):
        return MetricWriter._MetricRecordFactory(time, self._flush, mode = 'validation')

    def _flush(self, collection, time, mode):
        if self._use_tensorboard and mode == 'train':
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                        simple_value=value) for (tag, value) in collection])
            self._tensorboard_writer.add_summary(summary, time)
            self._tensorboard_writer.flush()

        for handler in self.handlers:
            handler.collect(collection, time, mode = mode)