import tensorflow as tf
import numpy as np
import os
# Warning: Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Generate data
x_data = np.float32(np.random.rand(2,100))
y_data = np.dot([0.100, 0.200], x_data) + 0.300

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1,2],-1.0,1.0,dtype=tf.float32))
y = tf.matmul(W,x_data) + b

loss = tf.reduce_mean(tf.square(y - y_data))
# tf.train.GradientDescentOptimizer(learn rate)
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
sess.run(train)

for step in range(0,201):
    sess.run(train)
    if step % 20 == 0:
        print('step=',step,'  W=',sess.run(W),'  b=',sess.run(b))




