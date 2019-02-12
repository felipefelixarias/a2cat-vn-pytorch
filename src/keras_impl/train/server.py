
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
                name='net', 
                max_queue_size = 100,
                **kwargs):
        self._config = kwargs
        self._batch_size = batch_size
        self._mode = mode
        self.training_q = Queue(maxsize=max_queue_size)
        self.prediction_q = Queue(maxsize=max_queue_size)
        self.stats = ProcessStats()

        self._agent = Agent(checkpoint_dir, 
            create_model, 
            device, 
            name = name,
            model_kwargs = dict(
                action_space_size = action_space_size,
                head = 'ac'))

        self._agent.initialize()
        #(self.stats.episode_count.value,) = self.agent.initialize()

        self._training_step = 0
        self._frame_counter = 0

        self._agents = []
        self._predictors = []
        self._trainers = []

    def add_agent(self):
        thread = ProcessAgent(self._mode, len(self._agents), self.prediction_q, self.training_q, self.stats.episode_log_q)
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
        thread = ThreadTrainer(self, len(self._trainers), batch_size = self._batch_size, **self._config)
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

    def main(self):
        self.stats.start()
        self.dynamic_adjustment.start()

        if Config.PLAY_MODE:
            for trainer in self.trainers:
                trainer.enabled = False

        learning_rate_multiplier = (
                                       Config.LEARNING_RATE_END - Config.LEARNING_RATE_START) / Config.ANNEALING_EPISODE_COUNT
        beta_multiplier = (Config.BETA_END - Config.BETA_START) / Config.ANNEALING_EPISODE_COUNT

        while self.stats.episode_count.value < Config.EPISODES:
            step = min(self.stats.episode_count.value, Config.ANNEALING_EPISODE_COUNT - 1)
            self.model.learning_rate = Config.LEARNING_RATE_START + learning_rate_multiplier * step
            self.model.beta = Config.BETA_START + beta_multiplier * step

            # Saving is async - even if we start saving at a given episode, we may save the model at a later episode
            if Config.SAVE_MODELS and self.stats.should_save_model.value > 0:
                self.save_model()
                self.stats.should_save_model.value = 0

            time.sleep(0.01)

        self.dynamic_adjustment.exit_flag = True
        while self.agents:
            self.remove_agent()
        while self.predictors:
            self.remove_predictor()
        while self.trainers:
            self.remove_trainer()