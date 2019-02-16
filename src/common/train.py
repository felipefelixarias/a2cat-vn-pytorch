import abc
import gym
import threading


class AbstractTrainer:
    def __init__(self, env_kwargs, model_kwargs):
        self.env = self._wrap_env(gym.make(**env_kwargs))
        self.model = self._create_model(**model_kwargs)
        pass

    def _wrap_env(self, env):
        return env

    @abc.abstractclassmethod
    def _create_model(self, **model_kwargs):
        pass

    @abc.abstractclassmethod
    def process(self, **kwargs):
        pass

    def run(self, process = None, **kwargs):
        if process is None:
            process = self.process
        if hasattr(self, '_run'):
            self._run(process = process, **kwargs)
        else:
            raise Exception('Run is not implemented')


class AbstractTrainerWrapper(AbstractTrainer):
    def __init__(self, trainer, *args, **kwargs):
        self.trainer = trainer
        self.unwrapped = trainer.unwrapped if hasattr(trainer, 'unwrapped') else trainer

    def _wrap_env(self, env):
        return self.trainer._wrap_env(env)

    def _create_model(self, **model_kwargs):
        return self.trainer._create_model(**model_kwargs)

    def process(self, **kwargs):
        return self.trainer.process(**kwargs)

    def _run(self, **kwargs):
        self.trainer.run(**kwargs)

    def stop(self, **kwargs):
        self.trainer.stop(**kwargs)


class SingleTrainer(AbstractTrainer):
    def __init__(self, env_kwargs, model_kwargs):
        super().__init__(env_kwargs = env_kwargs, model_kwargs = model_kwargs)
        self._global_t = None
        pass

    def _run(self, process):
        global_t = 0
        self._is_stopped = False
        while not self._is_stopped:
            tdiff, _, _ = process()
            global_t += tdiff

    def stop(self):
        self._is_stopped = True



class MultithreadTrainer(AbstractTrainer):
    class AgentThreadWrapper:
        def __init__(self, server, AgentProto, env_kwargs, model_kwargs):
            self._server = server
            self._agent_proto = AgentProto
            self._agent = None
            self._env_kwargs = env_kwargs
            self._model_kwargs = model_kwargs

        def __call__(self):
            if self._agent is None:
                self._agent = self._agent_proto(self._env_kwargs, self._model_kwargs)

            while not self._server._is_paused:
                tdiff, finished_episode_info = self._agent.process()
                self._server.process(_result = (tdiff, finished_episode_info))

    def process(self, _result):
        tdiff, _ = _result
        self._global_t += tdiff
        return _result

    def __init__(self, number_of_trainers, child_trainer, env_kwargs, model_kwargs):
        super(MultithreadTrainer, self).__init__(env_kwargs = env_kwargs, model_kwargs = model_kwargs)
        self._model_kwargs = model_kwargs
        self._env_kwargs = env_kwargs
        self._child_trainer = child_trainer
        self._number_of_trainers = number_of_trainers
        self._is_paused = False
        self._global_t = 0

    def _process(self):
        raise Exception('Not supported')

    def _run(self, process):
        self._agents = [MultithreadTrainer.AgentThreadWrapper(self, self._child_trainer, self._model_kwargs, self._env_kwargs) for _ in range(self._number_of_trainers)]
        self._train_threads = []
        for agent in self._agents:            
            thread = threading.Thread(target=agent)
            thread.setDaemon(True)
            self._train_threads.append(thread)          
            thread.start()