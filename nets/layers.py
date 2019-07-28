import tensorflow as tf
import numpy as np

def residual_layer(Input,filters):
    x = conv(Input, filters[0])
    x = conv(x,filters[1])
    x = tf.math.add(Input, x)
    return x

def conv(Input,filters,padding="SAME"):
    op_conv = tf.nn.conv2d(Input,filters,strides=1,padding=padding)
    return tf.nn.relu(op_conv)

def flatten(Input):
    shape = int(np.prod(Input.get_shape()[1:]))
    return tf.reshape(Input,[-1,shape])

def global_average_pooling(Input):
    x = tf.nn.avg_pool(Input, ksize=[1, 4, 4, 1024], strides=[1, 1, 1, 1], padding='VALID')
    return tf.reduce_mean(x, axis=[1, 2])

def Dense(Input,filters,activation=None):
    if activation=="relu":
      op_matmul=tf.matmul(Input,filters)
      return tf.nn.relu(op_matmul)
    return tf.matmul(Input,filters)
