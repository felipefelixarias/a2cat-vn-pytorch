from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input, Lambda,Concatenate, Reshape
import keras.backend as K
import tensorflow as tf
import numpy as np
import trfl

class BaseModel:
    def __init__(self, 
                    action_space_size,
                    image_size = (84, 84,),
                    head = 'dqn',
                    name = 'net',
                    device = None):
        self._name = name
        self._action_space_size = action_space_size
        self._image_size = image_size
        with tf.device(device):            
            self._initialize()

    def _initialize(self):
        raise Exception('Not implemented')

    def _build_net(self):
        # Inputs
        self.main_input = Input(shape=list(self._image_size) + [3], name="main_input")
        self.goal_input = Input(shape=list(self._image_size) + [3], name="goal_input")
        self.last_action_reward = Input(shape=(self._action_space_size + 1,), name = "last_action_reward")

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
        model = Concatenate(3)([image_stream, goal_stream])

        model = Conv2D(
            filters=32, #TODO: test 64
            kernel_size=[4,4],
            strides=[2,2],
            activation="relu",
            padding="valid",
            name="conv2")(model)

        model = Flatten()(model)
        
        model = Dense(
            units=256,
            activation="relu",
            name="fc3")(model)

        # model = Concatenate()([model, self.last_action_reward])
        return model

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

class ActorCriticModel(BaseModel):
    def __init__(self, gamma = 0.99, entropy_cost = 0.001, *args, **kwargs):
        self._gamma = gamma
        self._entropy_cost = entropy_cost
        self._rmsp_decay = 0.99
        self._rmsp_epsilon = 0.1
        self._gradient_clip_norm = 40.0

        super().__init__(*args, **kwargs)
        pass

    def _initialize(self):
        with tf.variable_scope('%s-net' % self._name):
            model = self._build_net()
            policy = Dense(
                units=self._action_space_size,
                activation="softmax",
                name="policy_fc"
            )(model)
            self.policy = policy

            value = Dense(
                units=1,
                name="value_fc"
            )(model)
            self.value = value

        self.inputs = [self.main_input, self.goal_input, self.last_action_reward]
        self.run_base_policy_and_value = K.Function(self.inputs, [self.policy, self.value])

        with tf.variable_scope('%s-optimizer' % self._name):
            self._build_loss((policy, value,))
            self._build_optimize()       


    def _build_loss(self, model):
        self.rewards = tf.placeholder(tf.float32, (None, 1))
        self.terminates = tf.placeholder(tf.bool, (None, 1))
        self.actions = tf.placeholder(tf.int32, (None, 1))
        gamma = tf.constant(self._gamma, dtype = tf.float32)
        entropy_cost = tf.constant(self._entropy_cost, dtype = tf.float32)
        self.bootstrap_value = tf.placeholder(tf.float32, (1,))
        
        (policy_logits, baseline_values) = model
        p_continues = (1.0 - tf.to_float(self.terminates)) * gamma
        a3c_loss, _ = trfl.sequence_advantage_actor_critic_loss(tf.reshape(policy_logits, [-1, 1, self._action_space_size]), 
                tf.reshape(baseline_values, [-1, 1]), 
                self.actions, 
                self.rewards, 
                p_continues,
                self.bootstrap_value,
                entropy_cost = entropy_cost)

        self.total_loss = tf.reduce_mean(a3c_loss, keepdims = False)

    def _build_optimize(self):
        self.learning_rate = tf.get_variable('learning_rate', tuple(), dtype = tf.float32, trainable=False)
        self.global_step = tf.get_variable('global_step', tuple(), dtype = tf.float32, trainable=False)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay = self._rmsp_decay, epsilon= self._rmsp_epsilon)
        
        # Compute and normalize gradients
        #trainables = tf.trainable_variables()
        grads_and_vars = self.optimizer.compute_gradients(self.total_loss)
        grads_and_vars = [(tf.clip_by_norm(grad, self._gradient_clip_norm), var) for grad, var in grads_and_vars]
        self.optimize_op = self.optimizer.apply_gradients(grads_and_vars, global_step = self.global_step)

        # Assign learning values op
        learning_rate_value = tf.placeholder(tf.float32, tuple())
        global_step_value = tf.placeholder(tf.float32,tuple())
        assign_learning_variables_op = [
            tf.assign(self.learning_rate, learning_rate_value),
            tf.assign(self.global_step, global_step_value)
        ]

        self.set_step = K.Function([learning_rate_value, global_step_value], assign_learning_variables_op)
        self.train_on_batch = K.Function(self.inputs + [self.actions, self.rewards, self.terminates, self.bootstrap_value], [self.total_loss], [self.optimize_op])


class DeepQModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def _initialize(self, model):
        adventage = Dense(
            units=self._action_space_size,
            activation=None,
            name="policy_fc"
        )(model)

        value = Dense(
            units=1,
            name="value_fc"
        )(model)

        model = Lambda(lambda val_adv: val_adv[0] + (val_adv[1] - K.mean(val_adv[1],axis=1,keepdims=True)),name="final_out")([value, adventage])
        self.model = Model(inputs = [self.main_input, self.goal_input, self.last_action_reward], outputs = model)
        self.model.compile("adam","mse")
        self.model.optimizer.lr = 0.0001
        
        self.model.train_function = self._make_train_function(self.model)
        self.predict = self._make_predict_function(self.model)

    def _make_predict_function(self, model):
        def predict(x):
            q = model.predict_on_batch(x)
            return np.argmax(q, axis=1)

        return predict

    def _make_train_function(self, model):
        if not hasattr(model, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')
        model._check_trainable_weights_consistency()
        if model.train_function is None:
            inputs = (model._feed_inputs +
                      model._feed_targets +
                      model._feed_sample_weights)
            if model._uses_dynamic_learning_phase():
                inputs += [K.learning_phase()]

            with K.name_scope('training'):
                with K.name_scope(model.optimizer.__class__.__name__):
                    training_updates = model.optimizer.get_updates(
                        params=model._collected_trainable_weights,
                        loss=model.total_loss)
                updates = (model.updates +
                           training_updates +
                           model.metrics_updates)
                # Gets loss and metrics. Updates weights at each call.
                return K.function(
                    inputs,
                    [model.total_loss] + model.metrics_tensors + model.outputs,
                    updates=updates,
                    name='train_function',
                    **model._function_kwargs)

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def save_weights(self, filename):
        self.model.save_weights(filename)

def create_model_fn(head, **kwargs):
    def model_fn(device, name):
        if head == 'dqn':
            return DeepQModel(device = device, name = name, **kwargs)

        elif head == 'ac':
            return ActorCriticModel(device = device, name = name, **kwargs)

        else:
            raise Exception('Head %s is unknown' % head)
    return model_fn

def create_model(head, device, name, **kwargs):
    return create_model_fn(head, **kwargs)(device, name)