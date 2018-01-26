import tensorflow as tf
import os
# Warning: Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


state = tf.Variable(0, name='counter')
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        result = sess.run([state])
        # result -> numpy ndarray
        print(result)
