import tensorflow as tf
import os


class MetricHandlerBase:
    def __init__(self, name, *args, **kwargs):
        self.name = name

class MetricWriter:
    class _MetricRecordFactory:
        def __init__(self, time, flush):
            self._flush = flush
            self.collection = []
            self._time = time

        def scalar(self, name, value):
            self.collection.append((name, value))
            return self

        def flush(self):
            self._flush(self.collection, self._time)
            self.collection.clear()
            return self

    def __init__(self, use_tensorboard = True, logdir = './logs'):
        self._use_tensorboard = use_tensorboard
        self._logdir = logdir

        if logdir is not None and len(logdir) > 0:
            if not os.path.exists(logdir):
                os.mkdir(logdir)

        if use_tensorboard:
            assert logdir is not None and len(logdir) > 0
            self._tensorboard_writer = tf.summary.FileWriter(logdir)
    
    def record(self, time):
        return MetricWriter._MetricRecordFactory(time, self._flush)

    def _flush(self, collection, time):
        if self._use_tensorboard:
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                        simple_value=value) for (tag, value) in collection])
            self._tensorboard_writer.add_summary(summary, time)
            self._tensorboard_writer.flush()