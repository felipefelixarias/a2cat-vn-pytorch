import random
import numpy as np
import tensorflow as tf
import functools
import os, sys

if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    sys.path.insert(0,parentdir) 

from model.model import UnrealModel
from environment.environment import Environment
from train.experience import ExperienceFrame



class ExperienceReplay():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        STATE_INDICES = [0,3]
        items = random.sample(self.buffer,size)
        def convert_dict(dicts, i):
            if len(dicts) == 0:
                return {}
            else:
                return {key:np.stack([y[i][key] for y in dicts], 0) for key in dicts[0][i].keys()}
        def convert(items, i):
            if i in STATE_INDICES:
                return convert_dict(items, i)
            return np.array([x[i] for x in items])

        batch = [convert(items, i) for i in range(5)]
        return batch


def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

class DoubleQLearning:
    def __init__(self, 
                model_fn, 
                env, 
                tau, 
                start_epsilon, 
                end_epsilon, 
                annealing_steps, 
                episode_length, 
                pre_train_steps,
                checkpoint_dir,
                update_frequency,
                num_episodes,
                gamma,
                device,
                batch_size,
                **kwargs):
        tf.reset_default_graph()
        self._env = env
        self._checkpoint_dir = checkpoint_dir
        self._episode_length = episode_length
        self._pre_train_steps = pre_train_steps
        self._gamma = gamma
        self._update_frequency = update_frequency
        self._num_episodes = num_episodes
        self._batch_size = batch_size
        self._start_epsilon = start_epsilon
        self._end_epsilon = end_epsilon
        self._annealing_steps = annealing_steps

        
        self._global_net = model_fn(name = 'global_net', device = device)
        self._global_net.prepare_loss()
        self._local_net = model_fn(name = 'local_net', device = device)

        with tf.device(device):
            trainables = tf.trainable_variables()
            self._update_target_ops = updateTargetGraph(trainables, tau)
            self._trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self._updateModel = self._trainer.minimize(self._global_net.total_loss)
            self._init = tf.global_variables_initializer()
        
        self._saver = tf.train.Saver()

    def save(self):
        # Save
        if not os.path.exists(self._checkpoint_dir):
            os.mkdir(self._checkpoint_dir)
    
        print('Start saving.')
        self._saver.save(self._sess,
                        self._checkpoint_dir + '/' + 'checkpoint',
                        global_step = self._global_t)
        print('End saving.')

    def _process_state(self, state, action = -1, reward = 0):
        return {'image': state['image'], 
            'goal': state['goal'], 
            'action_reward': ExperienceFrame.concat_action_and_reward(action, self._env.action_space.n, reward, state)}

    def run(self):
        epsilon = self._start_epsilon
        epsilonStepDrop = (self._start_epsilon - self._end_epsilon) / self._annealing_steps
        experienceBuffer = ExperienceReplay()

        #create lists to contain total rewards and steps per episode
        jList = []
        rList = []
        self._global_t = 0

        config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=False)
        with tf.Session(config = config) as sess:
            self._sess = sess
            sess.run(self._init)
            
            checkpoint = tf.train.get_checkpoint_state(self._checkpoint_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                self._saver.restore(sess, checkpoint.model_checkpoint_path)
                print("checkpoint loaded:", checkpoint.model_checkpoint_path)
                tokens = checkpoint.model_checkpoint_path.split("-")
                # set global step
                self._global_t = int(tokens[1])
                print(">>> global step set: ", self._global_t)
        
            else:
                print("Could not find old checkpoint")

            for i in range(self._num_episodes):
                episodeBuffer = ExperienceReplay()
                #Reset environment and get first new observation
                s = self._env.reset()
                s = self._process_state(s)
                d = False
                rAll = 0
                j = 0
                #The Q-Network
                while j < self._episode_length:
                    j+=1
                    #Choose an action by greedily (with e chance of random action) from the Q-network
                    if np.random.rand(1) < epsilon or self._global_t < self._pre_train_steps:
                        a = np.random.randint(0, self._env.action_space.n)
                    else:
                        a = sess.run(self._global_net.predict, 
                            feed_dict= {
                                self._global_net.base_input: [s['image']],
                                self._global_net.goal_input: [s['goal']],
                                self._global_net.base_last_action_reward_input: [s['action_reward']]
                            })[0]

                    s1,r,d,_ = self._env.step(a)
                    s1 = self._process_state(s1, a, r)
                    self._global_t += 1
                    episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.
            
                    if self._global_t > self._pre_train_steps:
                        if epsilon > self._end_epsilon:
                            epsilon -= epsilonStepDrop
                
                        if self._global_t % (self._update_frequency) == 0:
                            trainBatch = experienceBuffer.sample(self._batch_size)
                            #Below we perform the Double-DQN update to the target Q-values
                            Q1 = sess.run(self._global_net.predict,
                                feed_dict={
                                    self._global_net.base_input: trainBatch[3]['image'],
                                    self._global_net.goal_input: trainBatch[3]['goal'],
                                    self._global_net.base_last_action_reward_input: trainBatch[3]['action_reward']
                                })
                            Q2 = sess.run(self._local_net.q_out,
                                feed_dict={
                                    self._local_net.base_input:trainBatch[3]['image'],
                                    self._local_net.goal_input: trainBatch[3]['goal'],
                                    self._local_net.base_last_action_reward_input: trainBatch[3]['action_reward']
                                })

                            end_multiplier = -(trainBatch[4] - 1)
                            doubleQ = Q2[range(self._batch_size),Q1]
                            targetQ = trainBatch[2] + (self._gamma * doubleQ * end_multiplier)
                            #Update the network with our target values.
                            _ = sess.run(self._updateModel, feed_dict={
                                    self._global_net.base_input: trainBatch[0]['image'],
                                    self._global_net.goal_input: trainBatch[0]['goal'],
                                    self._global_net.base_last_action_reward_input: trainBatch[0]['action_reward'],
                                    self._global_net.target_q: targetQ, 
                                    self._global_net.actions: trainBatch[1]
                                })
                            
                            updateTarget(self._update_target_ops, sess) #Update the target network toward the primary network.

                    rAll += r
                    s = s1
                    
                    if d == True:
                        break
                
                experienceBuffer.add(episodeBuffer.buffer)
                jList.append(j)
                rList.append(rAll)
                #Periodically save the model. 
                if i % 1000 == 0:
                    self.save()

                if len(rList) % 10 == 0:
                    print(self._global_t,np.mean(rList[-10:]), epsilon)
            
            self.save()
            self._sess = None
        print("Percent of succesful episodes: " + str(sum(rList)/ self._num_episodes) + "%")
        pass

def get_options():
    tf.app.flags.DEFINE_integer('batch_size', 32, 'How many experiences to use for each training step.')
    tf.app.flags.DEFINE_integer('update_frequency', 4, 'How often to perform a training step.')
    tf.app.flags.DEFINE_float('gamma', .99, 'Discount factor on the target Q-values')
    tf.app.flags.DEFINE_float('start_epsilon', 1, 'Starting chance of random action')
    tf.app.flags.DEFINE_float('end_epsilon', 0.1, 'Final chance of random action')
    tf.app.flags.DEFINE_integer('annealing_steps', 10000, 'How many steps of training to reduce startE to endE.')
    tf.app.flags.DEFINE_integer('num_episodes', 10000, 'How many episodes of game environment to train network with.')
    tf.app.flags.DEFINE_integer('pre_train_steps', 10000, 'How many steps of random actions before training begins.')
    tf.app.flags.DEFINE_integer('episode_length', 50, 'The max allowed length of our episode.')
    tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoints", "checkpoint directory")
    tf.app.flags.DEFINE_float('tau', 0.001, 'Rate to update target network toward primary network')
    return tf.app.flags.FLAGS

class Application:
    def __init__(self, flags):
        self._flags = flags
        pass

    def run(self):
        
        device = "/gpu:0"
        env = Environment.create_environment('maze', 'gr')

        model_fn = lambda name, device: UnrealModel(
                env.get_action_size(),
                0,
                thread_index = name, # -1 for global
                use_lstm = False,
                use_pixel_change = False,
                use_value_replay = False,
                use_reward_prediction = False,
                use_deepq_network = True,
                use_goal_input = env.can_use_goal(),              
                pixel_change_lambda = .05,
                entropy_beta = .001,
                device = device,
            )

        learning = DoubleQLearning(model_fn, env.get_env(), device = device, **self._flags.flag_values_dict())
        learning.run()

if __name__ == '__main__':
    flags = get_options()
    tf.app.run(lambda _: Application(flags).run())