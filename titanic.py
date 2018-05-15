#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/13 16:05
# @Author  : yanbo
# @Site    : 
# @File    : titanic.py
# @Software: PyCharm
# @python version:
import pandas as pd
import tensorflow as tf
import numpy as np
import re
from sklearn.model_selection import train_test_split
titles = {'mr':1,
          'mrs':2,
          'mme':2,
          'ms':3,
          'miss':3,
          'mlle':3,
          'don':4,
          'sir':4,
          'jonkheer':4,
          'major':4,
          'col':4,
          'dr':4,
          'master':4,
          'capt':4,
          'dona':5,
          'lady':5,
          'countess':5,
          'rev':7,}
def get_title(name):
    if pd.isnull(name):
        return 'Null'
    title_search = re.search('([A-Za-z]+)\.',name)
    if title_search:
        return title_search.group(1).lower()
    else:
        return 'None'
data_root = './data/'
data = pd.read_csv(data_root+'titanic/train.csv')
data['Sex'] = data['Sex'].apply(lambda s:1 if s =='male' else 0)
#data = data.fillna(0)
mean_age = data['Age'].mean()
#print(data['Age'][data.Age.isnull()])
data['Age'][data.Age.isnull()] = mean_age
data['Title'] = data['Name'].apply(lambda name:titles.get(get_title(name)))
data['Honor'] = data['Title'].apply(lambda title:1 if title== 4 or title ==5 else 0)
dataset_x = data[['Sex','Age','Pclass','SibSp','Parch','Fare','Title','Honor']].values
#print(data['Age'][data.Age.isnull()])

data['Deceased'] = data['Survived'].apply(lambda s:int(not s))
dataset_y = data[['Deceased','Survived']].values

X_train,X_test,y_train,y_test = train_test_split(dataset_x,dataset_y,test_size = 0.2,random_state = 42)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,shape = [None,8],name = 'x')
    y = tf.placeholder(tf.float32,shape=[None,2],name = 'y')
with tf.name_scope('classifier'):
    w = tf.Variable(tf.random_normal([8,2]),name = 'weights')
    b = tf.Variable(tf.zeros([2]),name = 'bias')

    y_pred = tf.nn.softmax(tf.matmul(x,w)+b)

    tf.summary.histogram('weights',w)
    tf.summary.histogram('bias',b)
with tf.name_scope('cost'):
    cross_entropy = -tf.reduce_sum(y*tf.log(y_pred+1e-10),reduction_indices=1)

    cost = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss',cost)
train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
with tf.name_scope('accuracy'):
    correct = tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1))
    acc_op = tf.reduce_mean(tf.cast(correct,tf.float32))
    tf.summary.scalar('accuracy',acc_op)
    #存储模型
    #save_path = saver.save(sess,"./model.ckpt")
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter('./logs',sess.graph)
    merged = tf.summary.merge_all()
    for epoch in range(100):
        total_loss = 0
        for i in range(len(X_train)):
            feed = {x:[X_train[i]],y:[y_train[i]]}
            _,loss = sess.run([train_op,cost],feed_dict=feed)
            total_loss += loss
        print("epoch: %04d,total loss=%.9f" % (epoch + 1, total_loss))
        #每一步在训练集表现
        summary,accuracy = sess.run([merged,acc_op],feed_dict= {x:X_train,y:y_train})
        writer.add_summary(summary, epoch)
        # #在测试集表现
        # summary,accuracy = sess.run([merged,acc_op],feed_dict= {x:X_test,y:y_test})
        # writer.add_summary(summary,epoch)

    print('training done!')
    #tensorboard --logdir=./logs(cd到logs所在目录，注意不要多一个空格）

#测试存储功能
# with tf.Session() as sess1:
#     saver.restore(sess1,"model.ckpt")
#     pred = sess1.run(y_pred, feed_dict={x: X_test})
#     correct = np.equal(np.argmax(pred,1),np.argmax(y_test,1))
#     accuracy = np.mean(correct.astype(np.float32))
#     print("Accuracy on validation set：%.9f" % accuracy)

#读取测试文件，生成结果
def submit(saver):
    testdata = pd.read_csv('./data/titanic/test.csv')
    testdata = testdata.fillna(0)
    testdata['Sex'] = testdata['Sex'].apply(lambda x:1 if x == 'male' else 0)
    X_test = testdata[['Sex','Age','Pclass','SibSp','Parch','Fare']].as_matrix()
    with tf.Session() as sess2:
        saver.restore(sess2,'./model/model.ckpt')
        pred = np.argmax(sess2.run(y_pred, feed_dict={x: X_test}),1)
        submission = pd.DataFrame({"PassengerId":testdata["PassengerId"],
                                   "Survived":pred})
        submission.to_csv("titanic-submission.csv",index = False)


