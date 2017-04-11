import numpy as np

import tensorflow as tf

from model import iColorUNet
from data_utils import *



if __name__ == '__main__':
    batch_size = 16
    num_epochs = 10
    
    image_l, image_ab, dataset_info = read_imagenet_data(
        'data/train.txt', '/home/young/data/dataset/ILSVRC2012',
        batch_size, num_epochs=None, shuffle=False
    )
    
    data_ab = tf.zeros([batch_size, 224, 224, 2])
    data_mask = tf.zeros([batch_size, 224, 224, 1])
    
    unet = iColorUNet(image_l, data_ab, data_mask, image_ab)
    
    train_op = tf.train.AdamOptimizer().minimize(unet.loss)
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    sess = tf.Session()
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    sess.run(init_op)
    
    n_iterations = int(num_epochs * dataset_info.num_examples / batch_size)
    
    for iterations in xrange(n_iterations):
        loss = sess.run([unet.loss, train_op], {unet.is_training: True})[0]
        print 'Epoch: {} / {}, Batch: {} / {}, Loss: {}'.format(
            int(iterations * batch_size / dataset_info.num_examples) + 1,
            num_epochs,
            iterations % int(dataset_info.num_examples / batch_size) + 1,
            int(dataset_info.num_examples / batch_size),
            loss
        )
    
    coord.request_stop()
    
    coord.join(threads)
    sess.close()
    