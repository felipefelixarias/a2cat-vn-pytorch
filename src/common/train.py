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
    def _create_model(self, model_kwargs):
        pass

    @abc.abstractclassmethod
    def process(self, **kwargs):
        pass


class SingleTrainer(AbstractTrainer):
    def __init__(self, max_time_steps, env_kwargs, model_kwargs):
        super().__init__(env_kwargs = env_kwargs, model_kwargs = model_kwargs)
        self._global_t = None
        self.max_time_steps = max_time_steps
        pass

    def run(self):
        global_t = 0
        while global_t < self.max_time_steps:
            tdiff, _ = self.process()
            global_t += tdiff




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

    def run(self):
        self._agents = [MultithreadTrainer.AgentThreadWrapper(self, self._child_trainer, self._model_kwargs, self._env_kwargs) for _ in range(self._number_of_trainers)]
        self._train_threads = []
        for agent in self._agents:            
            thread = threading.Thread(target=agent)
            thread.setDaemon(True)
            self._train_threads.append(thread)          
            thread.start()
            