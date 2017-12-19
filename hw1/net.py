#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 22:40:02 2017

@author: wangweiguo
"""
import tensorflow as tf

class MLP:
    def __init__(self, X_dim, y_dim, hidden_layer_dims=[50,100]):
        self.hidden_layer_dims = hidden_layer_dims
        self.X_dim = X_dim
        self.y_dim = y_dim
        self.weights = {}
        self.build_graph()
        self.sess = tf.Session()
    def build_graph(self):
        self.X = tf.placeholder(tf.float32, shape=(None,self.X_dim))
        self.y = tf.placeholder(tf.float32, shape=(None,self.y_dim))
        
        # multi fullconnection
        self.net = self.X
        last_dim = self.X_dim
        for i, h in enumerate(self.hidden_layer_dims):
            print (i)
            self.net = self.full_connection(self.net, last_dim, h, 'fc_layer' + str(i), relu=True)
            last_dim = h
        
        print (last_dim, self.y_dim,)
        self.net = self.full_connection(self.net, last_dim, self.y_dim, 'pred', relu=False)

        self.pred = self.net
        self.loss = self.make_loss(self.net, self.y)

    def full_connection(self, net, input_dim, output_dim, name='', relu=True):
        print (input_dim, output_dim)
        W = tf.Variable(tf.truncated_normal([input_dim, output_dim]), name=name+'_W', )
        self.weights[name+'_W'] = W
        b = tf.Variable(tf.truncated_normal([output_dim]), name=name+'_b')
        self.weights[name+'_b'] = b

        net = tf.add(tf.matmul(net, W), b)

        if relu:
            net = tf.nn.relu(net)
        
        return net
    
    
    def make_loss(self, y_pred, y):
        loss = tf.nn.l2_loss(y_pred - y)
        tf.summary.scalar('l2_loss', loss)
        return loss
    


if __name__ == '__main__':
    mlp = MLP(11, 3)
