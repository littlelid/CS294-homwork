#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:40:08 2017

@author: wangweiguo
"""

import tensorflow as tf
import numpy as np
import gym
from net import MLP
from data import Data

class DAgger:
    def __init__(self, gym_env, net):
        self.use_dagger = False
        self.net = net
        self.data = Data(gym_env)
        self.gym_env = gym_env
        self.render = True

        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.net.loss)
        self.sess = tf.Session()

        try:    # for compatible
            self.summary_op = tf.summary.merge_all()
        except:
            self.summary_op = tf.merge_all_summaries()

        try:
            self.writer = tf.summary.FileWriter('./log')
        except:
            self.writer = tf.train.SummaryWriter('./log')

    def train(self,):
        self.sess.run(tf.initialize_all_variables())
        learning_rate = 1e-1
        for i in range(10000):
            #if learning_rate % 1000 == 0:
            #    learning_rate *= 0.5
            batch_X, batch_y = self.data.next_batch()
            feed_dict = {self.net.X:batch_X, self.net.y:batch_y, self.learning_rate:learning_rate}
            summary, loss, _ = self.sess.run([self.summary_op, self.net.loss, self.optimizer], feed_dict=feed_dict)
            print(loss)
            self.writer.add_summary(summary, i)


    def test(self):
        env = self.gym_env
        policy_fn = self.predict
        max_test_steps = 1000
        steps = 0
        totalr = 0.
        while True:
            obs = env.reset()
            done = False


            while not done:
                action = policy_fn(obs[None, :])
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1

                if self.render:
                    env.render()

                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_test_steps))
            if steps >= max_test_steps:
                break
        print('total_reward:', totalr)

    def predict(self, X):
        y_pred = self.sess.run(self.net.pred, feed_dict={self.net.X:X})
        return y_pred

    def save(self,):
        pass
    
        
if __name__ == '__main__':
    #dagger = DAgger()
    env = gym.make('Hopper-v1')
    X_dim = env.observation_space.shape[0]
    y_dim = env.action_space.shape[0]
    net = MLP(X_dim, y_dim, [50, 100])
    dagger = DAgger(env, net)
    dagger.train()
    dagger.test()


