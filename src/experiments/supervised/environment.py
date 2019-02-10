import math
import numpy as np
import train.experience
import random

def _compute_optimal_policy_reward(distance, reward_config, gamma):
    if gamma < 1:
        return reward_config[0] * math.pow(gamma, distance) + reward_config[1] * (1 - math.pow(gamma, distance)) / (1 - gamma)
    else:
        return reward_config[0] + reward_config[1] * distance

def create_dataset(env, gamma, use_deterministic = True):
    OBS_PER_CONF = 100
    NUMBER_OF_CONF = 100
    for _ in range(NUMBER_OF_CONF):
        env.reset()
        for _ in range(OBS_PER_CONF):
            while env.last_state['optimal_distance'] < 2:
                env.reset_start()

            old_state = env.last_state
            last_action = env.last_state['optimal_action']
            if not np.any(last_action):
                last_action = np.ones_like(last_action)

            if use_deterministic:
                # If we use deterministic policy
                # We will always take the first action available
                (act_idx,) = np.where(last_action == 1)
                last_action[:] = 0
                last_action[act_idx[0]] = 1
            else:
                last_action = last_action / np.sum(last_action)

            (optimal_actions,) = np.where(last_action)


            old_action = random.choice(optimal_actions)

            _, old_reward, _, _ = env.process(old_action)
            old_reward = np.clip(old_reward, -1, 1)
            last_action_frame = train.experience.ExperienceFrame.concat_action_and_reward(
                old_action,
                env.get_action_size(),
                old_reward,
                old_state)

            optimal_action = env.last_state['optimal_action']
            if use_deterministic:
                # If we use deterministic policy
                # We will always take the first action available
                (act_idx,) = np.where(optimal_action == 1)
                optimal_action[:] = 0
                optimal_action[act_idx[0]] = 1
            else:
                optimal_action = optimal_action / np.sum(optimal_action)


            reward = _compute_optimal_policy_reward(env.last_state['optimal_distance'], env.reward_configuration, gamma)
            datarow = (env.last_state['image'], env.last_state['goal'], optimal_action, reward, last_action_frame)
            yield datarow