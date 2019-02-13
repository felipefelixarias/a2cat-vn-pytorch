import tensorflow as tf
from collections import namedtuple
from environment.environment import create_env


class Predictor:
    def __init__(self, request_queue, response_queue):
        self._request_queue = request_queue
        self._response_queue = response_queue

    def predict(self, state):
        self._request_queue.put(state)
        return self._response_queue.get()

class AgentProcess:
    def __init__(self, mode, id,  predictor, training_queue, env_args):
        self.exit_flag = False
        self._env_args = env_args
        self._env = None
        self._predictor = predictor
        self._training_queue = training_queue
        self._mode = mode
        self.id = id
        pass

    def run(self):
        self._env = create_env(self._env_args)
        time.sleep(np.random.rand())        
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        while not self.exit_flag:
            self._process_episode()

    def _select_action(self, prediction):
        if self._mode == 'eval':
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self.actions, p=prediction)
        return action

    def _process_episode(self):
        state = self._env.reset()
        done = False

        episode_data = []
        episode_end = None
        while not self.exit_flag and not done:
            # Do one env step
            (policy, value,) = self._predictor.predict(state)
            action = self._select_action(policy)
            

            self._training_queue.



class Trainer:
    def __init__(self, model, gamma = 0.99, entropy_cost = 0.001):
        self._gamma = gamma
        self._entropy_cost = entropy_cost
        self._rmsp_decay = 0.99
        self._rmsp_epsilon = 0.1
        self._gradient_clip_norm = 40.0
        self._build_graph(model)
        pass

    def _build_graph(self, model):
        self._build_loss(model)
        pass

    def _build_loss(self, model):
        self.rewards = tf.placeholder(tf.float32, (None, 1))
        self.terminates = tf.placeholder(tf.int8, (None, 1))
        self.actions = tf.placeholder(tf.int8, (None, 1))
        gamma = tf.constant(self._gamma, dtype = tf.float32)
        entropy_cost = tf.constant(self._entropy_cost, dtype = tf.float32)
        self.bootstrap_value = tf.placeholder(tf.float32, (1,))
        

        (policy_logits, baseline_values) = model
        p_continues = (1.0 - tf.to_float(self.terminates)) * gamma
        a3c_loss = trfl.sequence_advantage_actor_critic_loss(policy_logits, 
                baseline_values, 
                self.actions, 
                self.rewards, 
                p_continues,
                self.bootstrap_value,
                entropy_cost = entropy_cost)

        self.total_loss = a3c_loss

    def _build_optimize(self, loss):
        self.learning_rate = tf.placeholder(tf.float32, tuple())
        self.global_step = tf.placeholder(tf.int32, tuple())
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay = self._rmsp_decay, epsilon= self._rmsp_epsilon)
        
        # Compute and normalize gradients
        gradients = self.optimizer.compute_gradients(loss)
        gradients, _ = tf.clip_by_global_norm(gradients, self._gradient_clip_norm)
        self.optimize_op = self.optimizer.apply_gradients(gradients, global_step = self.global_step)
        
        






