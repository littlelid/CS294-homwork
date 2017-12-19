#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:40:08 2017

@author: wangweiguo
"""

import tensorflow as tf
import numpy as np
import gym
#from net import MLP
from data import Data

class DAgger:
    def __init__(self, gym_env, net):
        self.use_dagger = False
        self.net = net
        self.data = Data(gym_env)
        self.gym_env = gym_env


        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.net.loss)
        self.summary_op = tf.summary.merge_all()
        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter('./log')

    def train(self,):
        learning_rate = 1e-1
        for i in range(100000):
            batch_X, batch_y = self.data.next_batch()
            feed_dict = {self.net.X:batch_X, self.net.y:batch_y, self.learning_rate:learning_rate}
            summary, loss, _ = self.sess.run([self.summary_op, self.net.loss, self.optimizer], feed_dict=feed_dict)
            self.writer.add_summary(summary, i)

    def test(self):
        max_test_step = 1000;
        while True:
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None, :])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)








    def save(self,):
        pass
    
        
