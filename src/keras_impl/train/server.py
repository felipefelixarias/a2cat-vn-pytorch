
from multiprocessing import Process, Queue, Value

import time
from keras_impl.train.threads import ThreadTrainer, ThreadPredictor
from keras_impl.train.processes import ProcessAgent, ProcessStats
from keras_impl.agent import Agent
from model.model_keras import create_model

class Server:
    def __init__(self,
                mode,
                action_space_size, 
                device, 
                checkpoint_dir, 
                batch_size, 
                min_training_batch_size,
                logdir,
                save_frequency, 
                print_frequency,
                learning_rate,
                beta,
                total_episodes,
                name='net', 
                max_queue_size = 100,
                **kwargs):
        self._config = kwargs
        self._batch_size = batch_size
        self._mode = mode
        self._min_training_batch_size = min_training_batch_size
        self._learning_rate = learning_rate
        self._total_episodes = total_episodes
        self._beta = beta
        self.training_q = Queue(maxsize=max_queue_size)
        self.prediction_q = Queue(maxsize=max_queue_size)
        self.stats = ProcessStats(logdir, save_frequency, print_frequency)

        self._agent = Agent(checkpoint_dir, 
            create_model, 
            device,
            learning_rate = learning_rate,
            beta = beta,
            name = name,
            model_kwargs = dict(
                beta = beta,
                head = 'ac'))

        self._agent.initialize()
        #(self.stats.episode_count.value,) = self.agent.initialize()

        self._training_step = 0
        self._frame_counter = 0

        self._agents = []
        self._predictors = []
        self._trainers = []

    def add_agent(self):
        thread = ProcessAgent(self._mode, len(self._agents), self.prediction_q, self.training_q, self.stats.episode_log_q, **self._config)
        self._agents.append(thread)
        thread.start()

    def remove_agent(self):
        thread = self._agents.pop()
        thread.exit_flag.value = True
        thread.join()

    def add_predictor(self):
        thread = ThreadPredictor(self, len(self._predictors), self._batch_size)
        self._predictors.append(thread)
        thread.start()

    def remove_predictor(self):
        thread = self._predictors.pop()
        thread.exit_flag = True
        thread.join()

    def add_trainer(self):
        thread = ThreadTrainer(self, len(self._trainers), min_training_batch_size = self._min_training_batch_size, **self._config)
        self._trainers.append(thread)
        thread.start()

    def remove_trainer(self):
        thread = self._trainers.pop()
        thread.exit_flag = True
        thread.join()

    def train_model(self, batch, trainer_id):
        (loss,) = self._agent.train(batch, trainer_id)
        self._training_step += 1
        self._frame_counter += batch.size

        self.stats.training_count.value += 1
        # TODO: add metrics

    def save_model(self):
        self.model.save(self.stats.episode_count.value)

    def put_prediction_frame(self, frame, thread_id):
        self._agents[thread_id].wait_q.put(frame)

    def _get_thread_num(self):
        return (32, 2, 2) if self._mode == 'train' else (0,0,1)

    def run(self):
        self.stats.start()

        (n_agents, n_train, n_pred) = self._get_thread_num()
        for _ in range(n_agents):
            self.add_agent()

        for _ in range(n_train):
            self.add_trainer()

        for _ in range(n_pred):
            self.add_trainer()

        (learning_rate_start, learning_rate_end) = self._learning_rate
        (beta_start, beta_end) = self._beta 
        learning_rate_multiplier = (learning_rate_end - learning_rate_start) / self._total_episodes
        beta_multiplier = (beta_end - beta_start) / self._total_episodes

        while self.stats.episode_count.value < self._total_episodes:
            step = self.stats.episode_count.value
            self._agent.learning_rate = learning_rate_start + learning_rate_multiplier * step
            self._agent.beta = beta_start + beta_multiplier * step

            # Saving is async - even if we start saving at a given episode, we may save the model at a later episode
            if self.stats.should_save_model.value > 0:
                self.save_model()
                self.stats.should_save_model.value = 0

            time.sleep(0.01)

        while self._agents:
            self.remove_agent()
        while self._predictors:
            self.remove_predictor()
        while self._trainers:
            self.remove_trainer()