#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/3 18:10
# @Author  : yanbo
# @Site    : 
# @File    : sin_pre.py
# @Software: PyCharm
# @python version:
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell
from tensorflow.contrib.rnn import static_rnn
from tensorflow.python.ops import math_ops

def build_data(n):#每个样本是一个正弦值序列x和序列下一个值y,n是序列长度
    #rnn的任务是预测正弦值序列的下一个值y
    xs = []
    ys = []
    for i in range(2000):#样本数
        k = random.uniform(1,50)
        x = [[np.sin(k+j)] for j in range(0,n)]
        y = [np.sin(k+n)]
        xs.append(x)
        ys.append(y)
    train_x = np.array(xs[0:1500])
    train_y = np.array(ys[0:1500])
    test_x = np.array(xs[1500:])
    test_y = np.array(ys[1500:])
    return (train_x,train_y,test_x,test_y)

length = 10
(train_x,train_y,test_x,test_y) = build_data(length)
print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)

def train():
    time_step_size = length
    vector_size = 1
    batch_size = 10
    test_size = 10
    x = tf.placeholder("float",[None,length ,vector_size])
    y = tf.placeholder("float",[None,1])
    w = tf.Variable(tf.random_normal([10,1],stddev = 0.01))#stddev为标准差
    b = tf.Variable(tf.random_normal([1],stddev=0.01))
    def seq_predict_model (x,w,b,time_step_size,vector_size):
        #x的shape是[batch_size,time_step_size,vector_size]
        x = tf.transpose(x,[1,0,2])#把x转化成[time_step_size，batch_size,vector_size]
        x = tf.reshape(x,[-1,vector_size])#把x转化成[time_step_size*batch_size,vector_size]
        x = tf.split(x,time_step_size,0)#对第0维进行分割，分割大小为time_step_size,这样分割后，列表每一项是一个样本序列。
        cell = BasicRNNCell(num_units=10,activation=math_ops.tanh)#num_units是一个rnn单元中的输出类别数，比如现在是10，
        #那么比如再接一层softmax，它的输入个数就是10。同时这也就规定了rnn单元中的参数形状。

        initial_state = tf.zeros([batch_size,cell.state_size])#初始状态h0赋值为全0
        outputs,_states = static_rnn(cell,x,initial_state=initial_state)#outputs是每个时刻的输出，_states是最后的一个状态

        #线性激活
        return tf.matmul(outputs[-1],w)+b,cell.state_size#只取最后一个状态的输出，然后接一个线性激活输出。
    pred_y,_ = seq_predict_model(x,w,b,time_step_size,vector_size)

    loss = tf.square(tf.subtract(y,pred_y))#均方误差
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(50):
            for end in range(batch_size,len(train_x),batch_size):
                begin = end-batch_size
                x_value = train_x[begin:end]
                y_value = train_y[begin:end]
                sess.run(train_op,feed_dict={x:x_value,y:y_value})

                test_indices = np.arange(len(test_x))
                np.random.shuffle(test_indices)
                test_indices = test_indices[0:test_size]
                x_value = test_x[test_indices]
                y_value = test_y[test_indices]

                val_loss = np.mean(sess.run(loss,feed_dict={x:x_value,y:y_value}))
                print("Run %s" % i ,val_loss)
train()


