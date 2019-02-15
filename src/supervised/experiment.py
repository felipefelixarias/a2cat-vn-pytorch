if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)

import numpy as np
from supervised import maze_utils as utils
from keras.layers import Input, Dense, Conv2D, Concatenate, Flatten, BatchNormalization
from keras.models import Model, model_from_json
from keras import optimizers
import keras
from common.abstraction import AbstractAgent

def create_model(actions):
    image_stream = Input(shape=list((84,84,)) + [3], name="main_input")
    goal_stream = Input(shape=list((84,84,)) + [3], name="goal_input")
    inputs = [image_stream, goal_stream]

    # Basic network
    block1 = Conv2D(
        filters=32,
        kernel_size=[8,8],
        strides=[4,4],
        activation="relu",
        padding="valid",
        name="conv1")

    model = Concatenate(3)([image_stream, goal_stream])
    model = block1(model)

    model = Conv2D(
        filters=32, #TODO: test 64
        kernel_size=[4,4],
        strides=[2,2],
        activation="relu",
        padding="valid",
        name="conv2")(model)
    model = main_input
    model = Conv2D(
        filters = 32,
        kernel_size = [12,12],
        strides = [12,12],
        activation = 'relu',
        name = 'conv1'
    )(model)

    model = Flatten()(model)

    model = Dense(
        units=256,
        activation="relu",
        name="fc3")(model)

    model= Dense(
        units=actions,
        activation="softmax",
        name="fc4")(model)

    model = Model(inputs = inputs, outputs = [model])
    optimizer = optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

class SupervisedAgent(AbstractAgent):
    def __init__(self, deterministic = True):
        name = f"supervised-{'deterministic' if deterministic else 'stochastic'}"
        super().__init__(name)

        self.model = model_from_json(open('./checkpoints/%s-model.json' % name, 'r').read())
        self.model.load_weights('./checkpoints/%s-model.h5' % name)
        self.model.summary()

    def act(self, state):
        input_data = [[state['observation']]]#, [state['desired_goal']]]
        return np.argmax(self.model.predict(input_data)[0])

class ShortestPathAgent(AbstractAgent):
    def __init__(self):
        super().__init__('shortest-path')
        
        self.env = None
        self._actions = None
        self._goal = None

    def wrap_env(self, env):
        self.env = env
        if self.env.unwrapped is not None:
            self.env = self.env.unwrapped
        return env

    def act(self, state):
        if self._actions is None or self._goal != self.env._goal_pos:
            self._actions, _ = utils.build_graph_from_env(self.env)

        return np.argmax(self._actions[self.env._pos[0], self.env._pos[1], :])



if __name__ == '__main__':
    number_of_epochs = 100
    batch_size = 32
    deterministic = True
    data, labels = utils.Dataset(utils.build_multiple_goal_dataset(deterministic), batch_size).numpy()
    #data, labels = [np.repeat(x, 100, 0) for x in data], np.repeat(labels, 100, 0)

    labels = np.eye(1122)
    model = create_model(1122)
    model.summary()

    name = f"supervised-{'deterministic' if deterministic else 'stochastic'}-model"

    model.fit(x = data, y = labels, epochs=number_of_epochs, batch_size = batch_size)
    model.save_weights('./checkpoints/%s.h5' % name)
    json_string = model.to_json()
    with open('./checkpoints/%s.json' % name, 'w+') as f:
        f.write(json_string)
        f.flush()

