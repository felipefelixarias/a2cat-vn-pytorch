import abc
import gym
import threading
import os

class AbstractTrainer:
    def __init__(self, env_kwargs, model_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.env = None
        self._env_kwargs = env_kwargs
        self.model = None
        self._model_kwargs = model_kwargs
        self.name = 'trainer'

        self.is_initialized = False
        pass

    def save(self, path):
        model = self.model           
        model.save_weights(path + '/%s-weights.h5' % self.name)
        with open(path + '/%s-model.json' % self.name, 'w+') as f:
            f.write(model.to_json())
            f.flush()

    def wrap_env(self, env):
        return env

    @abc.abstractclassmethod
    def _initialize(self, **model_kwargs):
        pass

    @abc.abstractclassmethod
    def process(self, **kwargs):
        pass

    def _run(self, process):
        raise Exception('Run is not implemented')

    def __repr__(self):
        return '<%sTrainer>' % self.name

    def compile(self, compiled_agent = None, **kwargs):
        if compiled_agent is None:
            compiled_agent = CompiledTrainer(self)

        def run_fn(**kwargs):
            if not hasattr(self, '_run'):
                raise Exception('Run is not implemented')

            if isinstance(self._env_kwargs, dict):
                env = gym.make(**self._env_kwargs)
            else:
                env = self._env_kwargs
            self.env = self.wrap_env(env)
            self.model = self._initialize(**self._model_kwargs) 
            return self._run(compiled_agent.process)
            
        compiled_agent.run = run_fn
        return compiled_agent


class AbstractTrainerWrapper(AbstractTrainer):
    def __init__(self, trainer, *args, **kwargs):
        self.trainer = trainer
        self.unwrapped = trainer.unwrapped if hasattr(trainer, 'unwrapped') else trainer
        self.summary_writer = trainer.summary_writer if hasattr(trainer, 'summary_writer') else None

    def process(self, **kwargs):
        return self.trainer.process(**kwargs)

    def stop(self, **kwargs):
        self.trainer.stop(**kwargs)

    def save(self, path):
        self.trainer.save(path)

    def compile(self, compiled_agent = None, **kwargs):
        if compiled_agent is None:
            compiled_agent = CompiledTrainer(self)
        compiled = self.trainer.compile(compiled_agent = compiled_agent, **kwargs)
        if hasattr(self, 'run'):
            old_run = compiled.run
            def run(*args, **kwargs):
                return self.run(old_run, *args, **kwargs)

            compiled.run = run
        return compiled
        

class CompiledTrainer(AbstractTrainerWrapper):
    def __init__(self, target, *args, **kwargs):
        super().__init__(target, *args, **kwargs)
        self.process = target.process

    def __repr__(self):
        return '<Compiled %s>' % self.trainer.__repr__()


class SingleTrainer(AbstractTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._global_t = None
        pass

    def _run(self, process):
        global_t = 0
        self._is_stopped = False
        while not self._is_stopped:
            tdiff, _, _ = process()
            global_t += tdiff

        return None

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
        return None