import os
import datetime

import numpy as np

import tensorflow as tf
from skimage.color import lab2rgb
from skimage.io import imsave

from model import iColorUNet
from data_utils import *


def clip_lab(image):
    l, a, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    e = 1e-3
    return np.stack(
        [np.clip(l, 0. + e, 100. - e),
         np.clip(a, -86.185 + e, 98.254 - e),
         np.clip(b, -107.863 + e, 94.482 - e)],
        axis=2
    ).astype(np.float64)


def create_logging_dir():
    experiment_date_host = datetime.datetime.today().strftime(
        '%Y-%m-%d_%H-%M-%S_{}'.format(os.uname()[1])
    )
    
    experiment_data_dir = 'output/{}'.format(experiment_date_host)
    
    os.makedirs(experiment_data_dir)
    return experiment_data_dir


def record_batch_image(ground_truth, reveal, prediction, fname):
    
    idx = np.random.randint(0, ground_truth.shape[0])
    
    ground_truth = ground_truth[idx, :, :, :]
    reveal = reveal[idx, :, :, :]
    prediction = prediction[idx, :, :, :]
    
    combined = np.concatenate([ground_truth, reveal, prediction], axis=1)
    combined = clip_lab(combined)
    combined_rgb = lab2rgb(combined)
    imsave(fname, combined_rgb)
    
    
def record_unet_output(ground_truth, reveal, prediction, log_dir, epoch, batch):
    fname = os.path.join(
        log_dir,
        '{}__{}.png'.format(epoch, batch)
    )
    record_batch_image(ground_truth, reveal, prediction, fname)
    


if __name__ == '__main__':
    batch_size = 16
    num_epochs = 10
    display_every_itr = 500
    
    log_dir = create_logging_dir()
    
    image_l, image_ab, revealed, dataset_info = read_imagenet_data(
        'data/train.txt', '/home/young/data/dataset/ILSVRC2012',
        batch_size, num_epochs=None, shuffle=False
    )
    
    
    unet = iColorUNet(image_l, image_ab, revealed)
    
    train_op = tf.train.AdamOptimizer().minimize(unet.loss)
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    sess = tf.Session()
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    sess.run(init_op)
    
    num_iterations = int(num_epochs * dataset_info.num_examples / batch_size)
    num_batches = int(dataset_info.num_examples / batch_size)
    
    for iterations in xrange(num_iterations):
        epoch_idx = int(iterations * batch_size / dataset_info.num_examples) + 1
        batch_idx = iterations % int(dataset_info.num_examples / batch_size) + 1
        
        if iterations % display_every_itr == 0:
            loss, ground_truth, reveal, prediction, _ = sess.run(
                [unet.loss, unet.groud_truth_lab, unet.reveal_lab, unet.prediction_lab, train_op],
                {unet.is_training: True}
            )
            record_unet_output(ground_truth, reveal, prediction, log_dir, epoch_idx, batch_idx)
            
        else:
            loss = sess.run([unet.loss, train_op], {unet.is_training: True})[0]
        print 'Epoch: {} / {}, Batch: {} / {}, Loss: {}'.format(
            epoch_idx,
            num_epochs,
            batch_idx,
            num_batches,
            loss
        )
        
    
    coord.request_stop()
    
    coord.join(threads)
    sess.close()
    