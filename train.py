import numpy as np

import tensorflow as tf

from model import iColorUNet
from data_utils import *



if __name__ == '__main__':
    batch_size = 16
    
    image_l, image_ab = read_imagenet_data(
        'data/train.txt', batch_size, num_epochs=10, shuffle=False
    )
    
    data_ab = tf.zeros([batch_size, 224, 224, 2])
    data_mask = tf.zeros([batch_size, 224, 224, 1])
    
    unet = iColorUNet(image_l, data_ab, data_mask, image_ab)
    
    train_op = tf.train.AdamOptimizer().minimize(unet.loss)
    
    init_op = tf.group(tf.global_variables_initializer(), tf.initialize_local_variables())
    
    sess = tf.Session()
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    sess.run(init_op)
    
    iterations = 0
    
    try:
        while not coord.should_stop():
            loss = sess.run([unet.loss, train_op])[0]
            print 'Iteration: {}, Loss: {}'.format(iterations, loss)
            iterations += 1
    
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    
    coord.join(threads)
    sess.close()
    