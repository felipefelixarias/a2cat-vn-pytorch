from common.abstraction import AbstractAgent
from common.env_wrappers import UnrealObservationWrapper
from model.model import UnrealModel
import tensorflow as tf
import numpy as np

class UnrealAgent(AbstractAgent):
    def __init__(self, action_space_size, path = './checkpoints', 
        use_goal = True, use_pixel_change = True, use_value_replay = True, use_reward_prediction = True,
        use_lstm = False,  **kwargs):

        base_name = 'unreal' if not use_lstm else 'unreal-lstm'
        if not use_pixel_change and not use_value_replay and not use_reward_prediction:
            base_name = 'a3c' if not use_lstm else 'a3c-lstm'

        super(UnrealAgent, self).__init__(base_name)

        self.model = UnrealModel(action_space_size, 0, -1, use_lstm = use_lstm,
                use_pixel_change = use_pixel_change,
                use_value_replay = use_value_replay,
                use_reward_prediction = use_reward_prediction,
                use_goal_input = use_goal,
                pixel_change_lambda = None,
                entropy_beta = None,
                device = '/cpu:0')

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(path)
        saver.restore(sess, checkpoint.model_checkpoint_path)
        self.sess = sess

    def reset_state(self):
        self.model.reset_state()
        
    def wrap_env(self, env):
        return UnrealObservationWrapper(env)

    def act(self, state):
        input_state = dict(
            image = state.get('observation'),
            goal = state.get('desired_goal')
        )

        policy, _ = self.model.run_base_policy_and_value(self.sess, input_state, state.get('last_action_reward'))
        return np.random.choice(len(policy), p=policy)