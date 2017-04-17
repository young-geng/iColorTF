import os
import datetime
import argparse
import time

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
        '%Y-%m-%d_%H-%M-%S'
    )
    
    experiment_data_dir = os.path.join('output', experiment_date_host)
    checkpoint_dir = os.path.join(experiment_data_dir, 'checkpoint')
    
    
    os.makedirs(experiment_data_dir)
    os.makedirs(checkpoint_dir)
    return experiment_data_dir, checkpoint_dir


def record_batch_image(ground_truth, reveal, prediction, mask, fname):
    
    idx = np.random.randint(0, ground_truth.shape[0])
    
    ground_truth = ground_truth[idx, :, :, :]
    gray_scale = ground_truth.copy()
    gray_scale[:, :, 1:] = 0
    reveal = reveal[idx, :, :, :]
    prediction = prediction[idx, :, :, :]
    mask = mask[idx, :, :, :] > 0
    
    reveal = reveal * mask
    
    combined_1 = np.concatenate([gray_scale, ground_truth], axis=1)
    combined_2 = np.concatenate([reveal, prediction], axis=1)
    
    combined = np.concatenate([combined_1, combined_2], axis=0)
    combined = clip_lab(combined)
    combined_rgb = lab2rgb(combined)
    imsave(fname, combined_rgb)
    
    
def record_unet_output(ground_truth, reveal, prediction, mask, log_dir, epoch, batch):
    fname = os.path.join(
        log_dir,
        '{}_{}.png'.format(epoch, batch)
    )
    record_batch_image(ground_truth, reveal, prediction, mask, fname)
    

def parse_args():
    parser = argparse.ArgumentParser(description='Train the iColor model.')
    parser.add_argument(
        '--batch_size', type=int, default=24,
        help='batch size'
    )
    parser.add_argument(
        '--num_epochs', type=int, default=10,
        help='number of epochs'
    )
    parser.add_argument(
        '--display_every_itr', type=int, default=300,
        help='save a image every N iterations'
    )
    parser.add_argument(
        '--checkpoint_every_itr', type=int, default=5000,
        help='save the model every N iterations'
    )
    parser.add_argument(
        '--data_root', type=str, required=True,
        help='root dir for image data'
    )
    parser.add_argument(
        '--image_list', type=str, required=True,
        help='a file containing the list of all image filenames'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    display_every_itr = args.display_every_itr
    checkpoint_every_itr = args.checkpoint_every_itr
    
    log_dir, checkpoint_dir = create_logging_dir()
    
    image_l, image_ab, revealed, dataset_info = read_imagenet_data(
        args.image_list, args.data_root,
        batch_size, num_epochs=None, shuffle=False
    )
    
    unet = iColorUNet(image_l, image_ab, revealed)
    
    train_op = tf.train.AdamOptimizer().minimize(unet.loss)
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    sess = tf.Session()
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    saver = tf.train.Saver()
    
    sess.run(init_op)
    
    num_iterations = int(num_epochs * dataset_info.num_examples / batch_size)
    num_batches = int(dataset_info.num_examples / batch_size)
    
    for iterations in xrange(num_iterations):
        epoch_idx = int(iterations * batch_size / dataset_info.num_examples) + 1
        batch_idx = iterations % int(dataset_info.num_examples / batch_size) + 1
        
        last_time = time.time()
        
        if iterations % display_every_itr == 0:
            loss, ground_truth, reveal, prediction, mask, _ = sess.run(
                [unet.loss, unet.groud_truth_lab, unet.reveal_lab,
                 unet.prediction_lab, unet.reveal_mask, train_op],
                {unet.is_training: True}
            )
            record_unet_output(
                ground_truth, reveal, prediction, mask,
                log_dir, epoch_idx, batch_idx
            )
            
        else:
            loss = sess.run([unet.loss, train_op], {unet.is_training: True})[0]
        print 'Epoch: {} / {}, Batch: {} / {}, Loss: {}, Batch time: {} ms'.format(
            epoch_idx,
            num_epochs,
            batch_idx,
            num_batches,
            loss,
            int((time.time() - last_time) * 1000.)
        )
        
        if iterations % checkpoint_every_itr == 0:
            saver.save(sess, os.path.join(checkpoint_dir, 'model_{}_{}.ckpt'.format(epoch_idx, batch_idx)))
        
    
    coord.request_stop()
    
    coord.join(threads)
    sess.close()
    