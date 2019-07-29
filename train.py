import tensorflow as tf
from nets.layers import *
from utils.preprocess_image import preprocess

NUM_OF_EPOCHS=10
BATCH_SIZE=128

def load_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    (x_train, y_train)=preprocess(x_train, y_train)
    (x_test, y_test)=preprocess(x_test, y_test)
    return [(x_train, y_train),(x_test, y_test)]

def get_weights():
    weights={
          'conv_0_1':tf.Variable(tf.random_normal([3,3,3,32],stddev=0.1)),
        
          'conv_1_1':tf.Variable(tf.random_normal([3,3,32,64],stddev=0.1)),
        
          'conv_2_1':tf.Variable(tf.random_normal([3,3,64,128],stddev=0.1)),
          'conv_2_2':tf.Variable(tf.random_normal([3,3,128,64],stddev=0.1)),
          'conv_2_3':tf.Variable(tf.random_normal([3,3,64,128],stddev=0.1)),
        
          'conv_3_1':tf.Variable(tf.random_normal([3,3,128,256],stddev=0.1)),
          'conv_3_2':tf.Variable(tf.random_normal([3,3,256,128],stddev=0.1)),
          'conv_3_3':tf.Variable(tf.random_normal([3,3,128,256],stddev=0.1)),
        
          'conv_4_1':tf.Variable(tf.random_normal([3,3,256,512],stddev=0.1)),
          'conv_4_2':tf.Variable(tf.random_normal([3,3,512,256],stddev=0.1)),
          'conv_4_3':tf.Variable(tf.random_normal([3,3,256,512],stddev=0.1)),
          'conv_4_4':tf.Variable(tf.random_normal([3,3,512,256],stddev=0.1)),
          'conv_4_5':tf.Variable(tf.random_normal([3,3,256,512],stddev=0.1)),
        
          'conv_5_1':tf.Variable(tf.random_normal([3,3,512,1024],stddev=0.1)),
          'conv_5_2':tf.Variable(tf.random_normal([3,3,1024,512],stddev=0.1)),
          'conv_5_3':tf.Variable(tf.random_normal([3,3,512,1024],stddev=0.1)),
          'conv_5_4':tf.Variable(tf.random_normal([3,3,1024,512],stddev=0.1)),
          'conv_5_5':tf.Variable(tf.random_normal([3,3,512,1024],stddev=0.1)),
          
          'conv_6_1':tf.Variable(tf.random_normal([1,1,1024,1000],stddev=0.1)),
        
          'fc':tf.Variable(tf.random_normal([1000,10],stddev=0.1),),
    }

def model_graph(x,weights):
    x=conv(x,weights['conv_0_1'])
    x=tf.nn.max_pool2d(x,ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],padding="VALID")
    
    x=conv(x,weights['conv_1_1'])
    x=tf.nn.max_pool2d(x,ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],padding="VALID")
    
    x=conv(x,weights['conv_2_1'])
    x=residual_layer(x,[weights['conv_2_2'],weights['conv_2_3']])
    x=tf.nn.max_pool2d(x,ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],padding="VALID")
    
    x=conv(x,weights['conv_3_1'])
    x=residual_layer(x,[weights['conv_3_2'],weights['conv_3_3']])
    x=tf.nn.max_pool2d(x,ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],padding="VALID")
    
    x=conv(x,weights['conv_4_1'])
    x=residual_layer(x,[weights['conv_4_2'],weights['conv_4_3']])
    x=residual_layer(x,[weights['conv_4_4'],weights['conv_4_5']])
    x=tf.nn.max_pool2d(x,ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],padding="VALID")
    
    x=conv(x,weights['conv_5_1'])
    x=residual_layer(x,[weights['conv_5_2'],weights['conv_5_3']])
    x=residual_layer(x,[weights['conv_5_4'],weights['conv_5_5']])
    
    x=conv(x,weights['conv_6_1'])
    
    x=global_average_pooling(x)
    
    x=Dense(x,weights['fc'])
    
    return x

def train():
    Image=tf.placeholder(tf.float32,[None,32,32,3])
    labels=tf.placeholder(tf.float32)

    [(x_train, y_train),(x_test, y_test)]=load_dataset()
    weights=get_weights()

    pred=model_graph(Image,weights)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=labels))
    optimiser = tf.train.AdamOptimizer().minimize(loss)

    correct = tf.equal(tf.math.argmax(pred,1),tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct,"float"))

    with tf.Session() as sess:
      
      sess.run(tf.global_variables_initializer())
      
      for i in range(NUM_OF_EPOCHS):
          start=0
          end=BATCH_SIZE
          epoch_accuracy=0
          for j in range(int(len(x_train)/BATCH_SIZE)):
            epoch_x= x_train[start:end]
            epoch_y= y_train[start:end]
            _,c=sess.run([optimiser,loss], feed_dict={Image: epoch_x,labels: epoch_y})
            epoch_accuracy+=accuracy.eval({Image:epoch_x, labels:epoch_y})
            start=end+1
            end=end+BATCH_SIZE
          print("Epoch "+str(i))
          print('Training Accuracy:',epoch_accuracy/int(len(x_train)/BATCH_SIZE))
          start=0
          end=BATCH_SIZE
          epoch_accuracy=0
          for j in range(int(len(x_test)/BATCH_SIZE)):
            epoch_x= x_test[start:end]
            epoch_y= y_test[start:end]
            epoch_accuracy+=accuracy.eval({Image:epoch_x, labels:epoch_y})
            start=end+1
            end=end+BATCH_SIZE
          print('Testing Accuracy:',epoch_accuracy/int(len(x_test)/BATCH_SIZE))