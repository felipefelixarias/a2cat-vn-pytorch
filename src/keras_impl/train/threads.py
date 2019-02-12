from threading import Thread
import numpy as np
from train.experience import ExperienceReplay

class ThreadTrainer(Thread):
    def __init__(self, server, id, min_training_batch_size, **kwargs):
        super(ThreadTrainer, self).__init__()
        self.setDaemon(True)

        self._id = id
        self._server = server
        self._batch_size = min_training_batch_size
        self.exit_flag = False

    def run(self):
        while not self.exit_flag:
            batch = []
            while len(batch) <= self._batch_size:
                batch.extend(self._server.training_q.get())
        
            batch_size = len(batch)
            batch = ExperienceReplay.create_batch(batch)
            batch.size = batch_size
            self._server.train_model(batch, self._id)


class ThreadPredictor(Thread):
    def __init__(self, server, id, batch_size, image_size = (84, 84,), **kwargs):
        super(ThreadPredictor, self).__init__()
        self.setDaemon(True)

        self._id = id
        self._server = server
        self._batch_size = batch_size
        self._image_size = image_size
        self.exit_flag = False

    def run(self):
        while not self.exit_flag:
            batch = []
            ids = []
            while len(batch) < self._batch_size and not self._server.prediction_q.empty():
                id, state = self._server.prediction_q.get()
                ids.append(id)
                batch.append(state)

            batch_size = len(batch)
            batch = ExperienceReplay.create_batch(batch)
            batch.size = batch_size
            p, v = self._server.agent.predict_p_and_v(batch)

            for i in range(batch_size):
                if ids[i] < len(self._server.agents):
                    self._server.put_prediction_frame((p[i], v[i],), ids[i])