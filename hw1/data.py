# -*- coding: utf-8 -*-


import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy


class Data:
    
    def __init__(self, env, epoch = 5):

        self.policy_fn = load_policy.load_policy('./experts/Hopper-v1.pkl')
        self.env = env
        self.max_steps = 1000
        self.batch_size = 100
        self.batch_index = 0
        self.init_orgin_dataset()
        self.epoch = epoch

    def init_orgin_dataset(self, num_rollouts=20):
        print('Buiding origin dataset...')
        with tf.Session():
            tf_util.initialize()
            observations = []
            actions = []

            for i in range(num_rollouts):
                print('\trollout: ' + str(i) +  '/' + str(num_rollouts) )
                obs = self.env.reset()
                done = False

                steps = 0

                while not done:
                    action = self.policy_fn(obs[None,:])
                    observations.append(obs)

                    actions.append(action[0])

                    obs, _, done, _ = self.env.step(action)
                    steps += 1
                    if steps >= self.max_steps:
                        break
            self.observations = np.array(observations)
            self.actions = np.array(actions)
        print('observations shape:', self.observations.shape, ', actions shape', self.actions.shape)
        print('building dataset complete!')

    def next_batch(self):
        batch_observations = self.observations[self.batch_size * self.batch_index : self.batch_size * (self.batch_index + 1)]
        batch_actions      = self.actions[self.batch_size * self.batch_index : self.batch_size * (self.batch_index + 1)]
        self.batch_index += 1

        new_epoch = False
        if self.batch_index >= len(self.observations) / self.batch_size:
            new_epoch = True
            self.batch_index = 0
            self.random_shuffle()

        return batch_observations, batch_actions, new_epoch

    def aggregate(self, observation_new, action_new):
        observation_new = np.array(observation_new)
        action_new = np.array(action_new)

        self.observations = np.concatenate((self.observations, observation_new))
        self.actions      = np.concatenate((self.actions, action_new))
        self.random_shuffle()


    def random_shuffle(self):
        assert len(self.observations) == len(self.actions)
        num = len(self.observations)
        rdn_idx = np.arange(num)
        np.random.shuffle(rdn_idx)

        self.observations = self.observations[rdn_idx]
        self.actions      = self.actions[rdn_idx]

    def generate_new_data(self, fn, num_rollouts):
        observations = []
        for i in range(num_rollouts):
            obs = self.env.reset()

            done = False

            steps = 0

            while not done:
                action = fn(obs[None, :])
                observations.append(obs)

                obs, _, done, _ = self.env.step(action)
                steps += 1
                if steps >= self.max_steps:
                    break
        actions = []
        with tf.Session():
            tf_util.initialize()
            for obs in observations:
                a = self.policy_fn(obs)
                actions.append(a)
        self.aggregate(observations, actions)




        
if __name__ == '__main__':
    pass

        

