from common.train import AbstractTrainerWrapper
import numpy as np
import os, math
from util.metrics import MetricWriter
from collections import defaultdict

class SaveWrapper(AbstractTrainerWrapper):
    def __init__(self, *args, model_directory = './checkpoints', saving_period = 10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_directory = model_directory
        self._last_save = 0
        self.saving_period = saving_period

    def process(self, **kwargs):
        res = self.trainer.process()
        (tdiff, _, _) = res
        self._last_save += tdiff

        if self._last_save >= self.saving_period:
            self._save()
            self._last_save = 0

        return res

    def _save(self):
        print('Saving')
        
        model = self.unwrapped.model
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
            
        model.save_weights(self.model_directory + '/%s-weights.h5' % self.unwrapped.name)
        with open(self.model_directory + '/%s-model.json' % self.unwrapped.name, 'w+') as f:
            f.write(model.to_json())
            f.flush()

    def run(self, **kwargs):
        try:
            super().run(**kwargs)
        except KeyboardInterrupt:
            self._save()
            raise

        self._save()

    def __repr__(self):
        return '<Save %s>' % repr(self.trainer)

class TimeLimitWrapper(AbstractTrainerWrapper):
    def __init__(self, max_time_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._global_t = 0
        self.max_time_steps = max_time_steps

    def process(self, **kwargs):
        tdiff, episode_end, stats = self.trainer.process(**kwargs)
        self._global_t += tdiff
        if self._global_t >= self.max_time_steps:
            self.trainer.stop()
        return (tdiff, episode_end, stats)

    def __repr__(self):
        return '<TimeLimit(%s) %s>' % (self.max_time_steps, repr(self.trainer))

class EpisodeNumberLimitWrapper(AbstractTrainerWrapper):
    def __init__(self, max_number_of_episodes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._number_of_episodes = 0
        self.max_number_of_episodes = max_number_of_episodes

    def process(self, **kwargs):
        tdiff, episode_end, stats = self.trainer.process(**kwargs)
        if episode_end is not None:
            self._number_of_episodes += 1
            if self._number_of_episodes >= self.max_number_of_episodes:
                self.trainer.stop()
        return (tdiff, episode_end, stats)

    def __repr__(self):
        return '<EpisodeNumberLimit(%s) %s>' % (self.max_number_of_episodes, repr(self.trainer))

class SummaryWriter:
    def __init__(self, writer):
        self.accumulatives = defaultdict(list)
        self.lastvalues = dict()
        self.cummulatives = defaultdict(lambda: 0)
        self.writer = writer
        self.window_size = 100

    def add_last_value_scalar(self, name, value):
        self.lastvalues[name] = value

    def add_scalar(self, name, value):
        self.accumulatives[name].append(value)

    def add_cummulative(self, name, value):
        self.cummulatives[name] += value

    def _format_number(self, number):
        if isinstance(number, int):
            return str(number)

        return '{:.3f}'.format(number)

    def summary(self, global_t):
        values = [('step', global_t)]
        values.extend((key, value) for key, value in self.lastvalues.items())
        values.extend((key, np.mean(x[-self.window_size:])) for key, x in self.accumulatives.items())
        values.extend((key, x) for key, x in self.cummulatives.items())
        return ', '.join('{}: {}'.format(key, self._format_number(val)) for key, val in values)

    def commit(self, global_t):
        metrics_row = self.writer \
            .record(global_t)

        for (key, val) in self.accumulatives.items():
            metrics_row = metrics_row.scalar(key, np.mean(val))

        for (key, value) in self.lastvalues.items():
            metrics_row = metrics_row.scalar(key, value)

        metrics_row = metrics_row.flush()
        self.lastvalues = dict()
        return metrics_row


class EpisodeLoggerWrapper(AbstractTrainerWrapper):
    def __init__(self, logging_period = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_t = 0
        self.logging_period = logging_period
        self.summary_writer = SummaryWriter(MetricWriter())

        self._global_t = 0
        self._episodes = 0
        self._data = []
        self._losses = []

    def compile(self, compiled_agent = None, **kwargs):
        compiled_agent = super().compile(compiled_agent = compiled_agent, **kwargs)
        old_process = compiled_agent.process
        def late_process(**kwargs):
            data = old_process(**kwargs)
            if self._log_t >= self.logging_period:
                print(self.summary_writer.summary(self._global_t))
                self.summary_writer.commit(self._global_t)
                self._log_t = 0
            return data

        compiled_agent.process = late_process
        return compiled_agent    

    def process(self, **kwargs):
        tdiff, episode_end, stats = self.trainer.process(**kwargs)
        self._global_t += tdiff
        if episode_end is not None:
            self._episodes += 1
            self._log_t += 1

            episode_length, reward = episode_end
            self.summary_writer.add_last_value_scalar('episodes', self._episodes)
            self.summary_writer.add_scalar('episode_length', episode_length)
            self.summary_writer.add_scalar('reward', reward)

        if stats is not None:
            if 'loss' in stats:
                self.summary_writer.add_scalar('loss', stats.get('loss'))

            if 'win' in stats:
                self.summary_writer.add_scalar('win_rate', float(stats.get('win')))
                self.summary_writer.add_cummulative('win_count', int(stats.get('win')))                    

        return (tdiff, episode_end, stats)

    def __repr__(self):
        return '<EpisodeLogger %s>' % repr(self.trainer)

    def log(self, stats):
        episode_length, reward = tuple(map(lambda *x: np.mean(x), *self._data))
        loss = np.mean(self._losses) if len(self._losses) > 0 else float('nan')
        self._data = []
        self._losses = []
        report = 'steps: {}, episodes: {}, reward: {:0.5f}, episode length: {}, loss: {:0.5f}'.format(self._global_t, self._episodes, reward, episode_length, loss)

        metrics_row = self.writer \
            .record(self._global_t) \
            .scalar('reward', reward) \
            .scalar('episode_length', episode_length)

        if stats is not None and 'epsilon' in stats:
            report += ', epsilon:{:0.3f}'.format(stats.get('epsilon'))
            metrics_row = metrics_row.scalar('epsilon', stats.get('epsilon'))

        metrics_row.flush()

        if not math.isnan(loss):
            metrics_row = metrics_row.scalar('loss', loss)

        print(report)

def wrap(trainer, max_number_of_episodes = None, max_time_steps = None, episode_log_interval = None):
    if max_time_steps is not None:
        trainer = TimeLimitWrapper(max_time_steps, trainer = trainer)

    if max_number_of_episodes is not None:
        trainer = EpisodeNumberLimitWrapper(max_number_of_episodes, trainer = trainer)

    if episode_log_interval is not None:
        trainer = EpisodeLoggerWrapper(episode_log_interval, trainer = trainer)

    return SaveWrapper(trainer)