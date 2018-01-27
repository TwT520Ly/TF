import tensorflow as tf

weights = tf.Variable(tf.random_normal([784,200],stddev=0.35),name='weights')
weights_new = tf.Variable(weights.initial_value,name='weights_new')
