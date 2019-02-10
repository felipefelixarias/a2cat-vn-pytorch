from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input, Lambda,Concatenate, Reshape
import keras.backend as K
import tensorflow as tf

class BaseModel:
    def __init__(self, 
                    action_space_size, 
                    device = None, 
                    image_size = (84, 84,),
                    head = 'dqn',
                    use_softmax = True):
        self.action_space_size = action_space_size
        with tf.device(device):            
            self._build_net(action_space_size, image_size, use_softmax)

    def _initialize(self, model):
        pass


    def _build_net(self, action_space_size, image_size, use_softmax):
        # Inputs
        self.main_input = Input(shape=list(image_size + (3,)), name="main_input")
        self.goal_input = Input(shape=list(image_size + (3,)), name="goal_input")
        self.last_action_reward = Input(shape=(action_space_size + 1,), name = "last_action_reward")

        # Basic network
        block1 = Conv2D(
            filters=32,
            kernel_size=[8,8],
            strides=[4,4],
            activation="relu",
            padding="valid",
            name="conv1")

        image_stream = block1(self.main_input)
        goal_stream = block1(self.goal_input)
        model = Concatenate(3)((image_stream, goal_stream,))

        model = Conv2D(
            filters=32, #TODO: test 64
            kernel_size=[4,4],
            strides=[2,2],
            activation="relu",
            padding="valid",
            name="conv2")(model)

        model = Flatten(model)
        
        model = Dense(
            units=256,
            activation="relu",
            name="fc3")(model)

        model = Concatenate()((model, self.last_action_reward,))
        self._initialize(model)

        '''model = Conv2D(
            filters=64,
            kernel_size=[3,3],
            strides=[1,1],
            activation="relu",
            padding="valid",
            name="fc1")(model)
            Conv2D(
            filters= final_layer_size,
            kernel_size=[7,7],
            strides=[1,1],
            activation="relu",
            padding="valid",
            name="conv4")(model)'''


class DeepQModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def _initialize(self, model):
        adventage = Dense(
            units=self.action_space_size,
            activation=None,
            name="policy_fc"
        )(model)

        value = Dense(
            units=1,
            name="value_fc"
        )(model)

        model = Lambda(lambda val_adv: val_adv[0] + (val_adv[1] - K.mean(val_adv[1],axis=1,keepdims=True)),name="final_out")([value, adventage])
        self.model = Model(inputs = [self.main_input, self.goal_input], outputs = model)
        self.model.compile("adam","mse")
        self.model.optimizer.lr = 0.0001