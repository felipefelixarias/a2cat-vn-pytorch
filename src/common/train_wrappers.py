from common.train import AbstractTrainerWrapper
import numpy as np

class SaveTrainerWrapper(AbstractTrainerWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def process(self, **kwargs):
        res = super().process(**kwargs)
        (tdiff, episode_end) = res

        return res

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

class EpisodeLoggerWrapper(AbstractTrainerWrapper):
    def __init__(self, logging_period = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_t = 0
        self.logging_period = logging_period

        self._global_t = 0
        self._episodes = 0
        self._data = []
        self._losses = []

    def process(self, **kwargs):
        tdiff, episode_end, stats = self.trainer.process(**kwargs)
        self._global_t += tdiff
        if episode_end is not None:
            self._episodes += 1
            self._log_t += 1
            self._data.append(episode_end)
            if stats is not None and 'loss' in stats:
                self._losses.append(stats.get('loss'))

            if self._log_t >= self.logging_period:
                self.log(stats)
                self._log_t = 0

        return (tdiff, episode_end, stats)

    def __repr__(self):
        return '<EpisodeLogger %s>' % repr(self.trainer)

    def log(self, stats):
        episode_length, reward = tuple(map(lambda *x: np.mean(x), *self._data))
        loss = np.mean(self._losses) if len(self._losses) > 0 else float('nan')
        self._data = []
        self._losses = []
        report = 'steps: {}, episodes: {}, reward: {:0.5f}, episode length: {}, loss: {:0.5f}'.format(self._global_t, self._episodes, reward, episode_length, loss)
        if stats is not None and 'epsilon' in stats:
            report += ', epsilon:{:0.3f}'.format(stats.get('epsilon'))

        print(report)

def wrap(trainer, max_number_of_episodes = None, max_time_steps = None, episode_log_interval = None):
    if max_time_steps is not None:
        trainer = TimeLimitWrapper(max_time_steps, trainer = trainer)

    if max_number_of_episodes is not None:
        trainer = EpisodeNumberLimitWrapper(max_number_of_episodes, trainer = trainer)

    if episode_log_interval is not None:
        trainer = EpisodeLoggerWrapper(episode_log_interval, trainer = trainer)

    return trainer