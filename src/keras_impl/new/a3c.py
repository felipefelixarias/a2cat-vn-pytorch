import tensorflow as tf
from collections import namedtuple
from environment.environment import create_env, get_action_space_size
import numpy as np
from multiprocessing import Queue, Process
from model import create_model
import time
import sys

class Predictor:
    def __init__(self, thread_id, action_space_size, request_queue, response_queue):
        self._request_queue = request_queue
        self._response_queue = response_queue
        self.action_space_size = action_space_size
        self._thread_id = thread_id

    def predict(self, state):
        self._request_queue.put((self._thread_id, state,))
        return self._response_queue.get()


class MultiprocessingExceptionWrapper:
    def __init__(self, queue):
        self._queue = queue

    def __enter__(self):
        return self

    def __exit__(self, typename, value, fb):
        import traceback
        self._queue.put(traceback.format_exc())

        return True

class AgentProcess(Process):
    def __init__(self, mode, id,  predictor, training_queue, error_queue, env_args):
        super().__init__(name = 'agent-%s' % str(id))
        daemon = True
        self.exit_flag = False
        self._env_args = env_args
        self._env = None
        self._training_queue = training_queue
        self._mode = mode
        self._max_subepisode_length = 20 # TODO: experiment
        self._max_episode_length = None
        self._error_queue = error_queue
        self._predictor = predictor
        self.id = id
        pass

    def run(self):
        with MultiprocessingExceptionWrapper(self._error_queue):
            self._env = create_env(**self._env_args)
            time.sleep(np.random.rand())        
            np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

            while not self.exit_flag:
                for batch in self._process_episode():
                    self._training_queue.put(batch)

    def _select_action(self, prediction):
        if self._mode == 'eval':
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self._predictor.action_space_size, p=prediction)
        return action

    def _create_batch(self, data, next_value):
        return (data, next_value)

    def _too_long(self, length):
        return self._max_episode_length is not None and self._max_episode_length <= length

    def _process_episode(self):
        state = self._env.reset()
        done = False

        total_episode_length = 0
        total_reward = 0
        sub_episode_data = []
        sub_episode_length = 0
        while not self.exit_flag and not done and not self._too_long(total_episode_length):
            # Do one env step
            (policy, value,) = self._predictor.predict(state)
            action = self._select_action(policy)
            old_state = state
            state, reward, done, _ = self._env.step(action)
            total_reward += reward
            sub_episode_length += 1
            sub_episode_data.append([old_state, action, reward, done])
            total_episode_length += 1

            if done or sub_episode_length >= self._max_subepisode_length:
                if not done:
                    (_, next_value,) = self._predictor.predict(state)
                else:
                    next_value = 0.0

                train_batch = self._create_batch(sub_episode_data, next_value)
                yield train_batch
                sub_episode_data = []
                sub_episode_length = 0

                #Context switch
                time.sleep(0.01)

        return (total_episode_length, total_reward)


class AgentThreadError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass


class Trainer:
    def __init__(self, total_steps, model_kwargs, num_agents = 32, device = '/gpu:0'):
        self._num_agents = num_agents
        self._eval_request_queue = Queue(maxsize = num_agents)
        self._eval_request_queue = Queue(maxsize = num_agents)
        self._eval_response_queues = [Queue(maxsize = 8) for _ in range(num_agents)]
        self._error_queue = Queue(maxsize = num_agents)
        self._training_queue = Queue(maxsize = num_agents)
        self._total_steps = total_steps

        self._eval_batch_size = num_agents // 2
        self._env_args = dict(env_type = 'maze', env_name = 'gr')
        self._mode = 'train'

        self._agents = []

        self._model = create_model(device = device, name = 'net', **model_kwargs)

    def _spawn(self):
        for i in range(self._num_agents):
            predictor = Predictor(i, get_action_space_size(**self._env_args), self._eval_request_queue, self._eval_response_queues[i])
            thread = AgentProcess(self._mode, i, predictor, self._training_queue, self._error_queue, self._env_args)

            self._agents.append(thread)
            thread.start()

    def _learning_rate(self, global_t):
        return 1

    def _handle_training(self, global_t):
        max_batch_size = 8
        batch_size = 0

        # TODO: implement batching!!
        while not self._training_queue.empty() and batch_size < max_batch_size:
            (data, lastval, t) = self._training_queue.get_nowait()
            self._model.set_step(self._learning_rate(global_t), global_t)
            self._model.train_on_batch(data + [lastval])
            batch_size += 1
            global_t += t

        return global_t

    def _handle_evaluation(self):
        max_batch_size = self._num_agents

        thread_ids = []
        requests = []
        batch_size = 0
        while not self._eval_request_queue.empty() and batch_size < max_batch_size:
            thread_id, request = self._eval_request_queue.get_nowait()
            thread_ids.append(thread_id)
            requests.append(request)
            batch_size += 1

        batch_request = list(map(list, zip(*requests)))

        if batch_size > 0:
            result = self._model.run_base_policy_and_value(batch_request)
            for i in range(len(thread_ids)):
                self._eval_response_queues[thread_ids[i]].put(*[x[i] for x in result])
   
    def _handle_statistics(self, global_t):
        # TODO: implement
        pass

    def _handle_errors(self):
        # TODO: implement
        if not self._error_queue.empty():
            err_info = self._error_queue.get()
            sys.stderr.write(err_info)
            sys.stderr.flush()
            raise AgentThreadError()
        pass

    def run(self):
        # Starts agent processes
        self._spawn()

        learning_rate = 1.0
        global_t = 0
        try:
            while global_t < self._total_steps:
                self._handle_errors()
                self._model.set_step([learning_rate, global_t])

                # We first handle all training requests
                self._handle_errors()
                global_t = self._handle_training(global_t)

                # Then we handle evaluation requests
                self._handle_errors()
                self._handle_evaluation()

                self._handle_errors()
                self._handle_statistics(global_t)

                self._handle_errors()

            # TODO: save model

            for agent in self._agents:
                agent.exit_flag = True

            for agent in self._agents:
                agent.join()


        except AgentThreadError:
            pass

        finally:
            for agent in self._agents:
                if agent.is_alive():
                    agent.terminate()