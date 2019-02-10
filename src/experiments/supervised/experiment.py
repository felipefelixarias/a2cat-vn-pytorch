import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir) 

import itertools
import numpy as np
import tensorflow as tf

from environment.environment import Environment
from model.model import UnrealModel
from experiments.supervised.environment import create_dataset
from options import get_options

USE_GPU = True # To use GPU, set True

# get command line args
flags = get_options("training")

Environment.set_log_dir(flags.log_dir)

class Application(object):
  def __init__(self):
    pass
  
  def build_loss(self, policy_output, value_output, action_target, value_target):
    policy_loss = tf.contrib.losses.softmax_cross_entropy(policy_output, action_target)
    value_loss = 0.5 * tf.contrib.losses.mean_squared_error(value_output, value_target)
    return policy_loss + value_loss

  def build_iterator(self):
    datagen = lambda: create_dataset(self.env, flags.gamma)
    dataset = tf.data.Dataset.from_generator(datagen, (tf.float32, tf.float32, tf.bool, tf.float32, tf.float32)).shuffle(1000).batch(16, True)
    return dataset.make_initializable_iterator()

  def save(self):
    if not os.path.exists(flags.checkpoint_dir):
      os.mkdir(flags.checkpoint_dir)
  
    print('Start saving.')
    self.saver.save(self.sess,
                    flags.checkpoint_dir + '/' + 'checkpoint',
                    global_step = self.global_t)
    print('End saving.')

  def run(self):
    device = "/cpu:0"
    if USE_GPU:
      device = "/gpu:0"
    
    self.global_t = 0
    
    self.stop_requested = False
    self.terminate_reqested = False
    self.env = Environment.create_environment(flags.env_type, flags.env_name)

    self.global_network = UnrealModel(self.env.get_action_size(),
        self.env.get_objective_size(),
        -1,
        use_lstm = False,
        use_pixel_change = False,
        use_value_replay = False,
        use_reward_prediction = False,
        use_goal_input = True,
        pixel_change_lambda = flags.pixel_change_lambda,
        entropy_beta = flags.entropy_beta,
        device = device)

    self.action_target = tf.placeholder(tf.int32, (None,), 'action_target')
    self.reward_target = tf.placeholder(tf.float32, (None,), 'reward_target')
    self.loss = self.build_loss(self.global_network.base_pi_without_softmax, self.global_network.base_v, self.action_target, self.reward_target)
    self.trainers = []
    
    grad_applier = tf.train.GradientDescentOptimizer(0.001)
    self.apply_gradients = grad_applier.minimize(self.loss)
    
    # prepare session
    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)
    
    self.sess.run(tf.global_variables_initializer())
    
    # summary for tensorboard
    tf.summary.scalar("loss", self.loss)
    
    self.summary_op = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter(flags.log_dir,
                                                self.sess.graph)
    
    # init or load checkpoint with saver
    self.saver = tf.train.Saver(self.global_network.get_vars(), max_to_keep=0)
    
    checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
      self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
      print("checkpoint loaded:", checkpoint.model_checkpoint_path)
      tokens = checkpoint.model_checkpoint_path.split("-")
      # set global step
      self.global_t = int(tokens[1])
      print(">>> global step set: ", self.global_t)
      # set wall time
      wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)
      with open(wall_t_fname, 'r') as f:
        self.wall_t = float(f.read())
        self.next_save_steps = (self.global_t + flags.save_interval_step) // flags.save_interval_step * flags.save_interval_step
        
    else:
      print("Could not find old checkpoint")
      # set wall time
      self.wall_t = 0.0
      self.next_save_steps = flags.save_interval_step
    
    iterator = self.build_iterator()
    sample = iterator.get_next()

    epochs = 1
    while epochs <= 100:
        print('epoch %s started' % epochs)
        self.sess.run(iterator.initializer)
        total_loss = 0
        total_iter = 0
        while True:
            try:
                data = self.sess.run(sample)
                (input, goal, policy, reward, last_act) = data
                (loss_val, _, summary_str) = self.sess.run((self.loss, self.apply_gradients, self.summary_op), feed_dict= {
                    self.global_network.base_input: input,
                    self.global_network.goal_input: goal,
                    self.global_network.base_last_action_reward_input: last_act,
                    self.action_target: policy,
                    self.reward_target: reward,
                })
                
                self.summary_writer.add_summary(summary_str, self.global_t)
                self.summary_writer.flush()
                self.global_t += 1
                total_iter += 1
                total_loss += loss_val

            except tf.errors.OutOfRangeError:
                break
        print('loss is %f' % ((total_loss / total_iter)))
        
        epochs += 1
    self.save()

if __name__ == '__main__':
    Application().run()