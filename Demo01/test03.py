import tensorflow as tf
import os
# Warning: Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def fetch():
    input1 = tf.Variable(0.1)
    input2 = tf.Variable(0.2)
    input3 = tf.Variable(0.3)
    intermed = tf.add(input2, input3)
    mul = tf.multiply(input1, intermed)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        result = sess.run([mul, intermed])
        print(result)
        print([mul, intermed])

def feed():
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        result = sess.run([output], feed_dict={input1:[7.], input2:[5.]})
        print(result)
# output带有[]表示tensor，会显示[array([ 35.], dtype=float32)]
# output不带[]表示数据ndarray，会显示[35.]
# 对于输入，[7.]和7.是一样的
if __name__ == '__main__':
    feed()
