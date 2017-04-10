import numpy as np
import skimage.color

import tensorflow as tf


def read_image_path(image_path_file):
    image_path = []
    image_labels = []
    with open(image_path_file) as fin:
        for line in fin:
            ss = line.split()
            if len(ss) < 2:
                continue
            image_path.append(ss[0])
            image_labels.append(int(ss[1]))
    return image_path, image_labels
        

def load_image_from_path(input_queue):
    image_label = input_queue[1]
    image_array = tf.image.decode_jpeg(tf.read_file(input_queue[0]), channels=3)
    image_array = tf.image.resize_images(image_array, [224, 224])
    image_array = tf.cast(image_array, tf.uint8)
    image_l, image_ab = split_lab(rgb2lab(image_array))
    return image_l, image_ab
    
    
def batch_rgb2lab(x):
    # x: N x H x W x 3 or H x W x 3
    if len(x.shape) == 3:
        x = skimage.color.rgb2lab(x)
    else:
        x = np.swapaxes(x, 0, 2)
        x = skimage.color.rgb2lab(x)
        x = np.swapaxes(x, 0, 2)
    return x.astype(np.float32)
    
    
def rgb2lab(x):
    ret_val = tf.py_func(batch_rgb2lab, [x], tf.float32, stateful=False)
    ret_val.set_shape(x.get_shape())
    return ret_val
    

def split_lab(x):
    if len(x.shape) == 3:
        l = tf.slice(x, [0, 0, 0], [-1, -1, 1])
        ab = tf.slice(x, [0, 0, 1], [-1, -1, 2])
    else:
        l = tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, 1])
        ab = tf.slice(x, [0, 0, 0, 1], [-1, -1, -1, 2])
    return l, ab
    
    
def random_reveal_mask(image_ab):
    # image_ab: H x W x 2
    p_numpatch = 0.125 # probability for number of patches to use drawn from geometric distribution
    p_min_size = 0 # half-patch min size
    p_max_size = 4 # half-patch max size
    p_std = .25 # percentage of image for std where patch is located
    self.p_whole = .01
    
    pass
    
    
def random_reveal(image_ab):
    """ Randomly reveal patches in ab channels."""
    mask = tf.py_func(random_reveal_python, [image_ab], tf.float32, stateful=False)
    mask.set_shape(image_ab.shape[0], image_ab.shape[1], 1)
    reveal_ab = mask * image_ab
    return mask, reveal_ab


def read_imagenet_data(path, batch_size, num_epochs, shuffle=True):
    
    # Performing all the async data preprocessing on CPU.
    with tf.device('/cpu:0'):
        image_list, label_list = read_image_path(path)
        
        images = tf.convert_to_tensor(image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
        
        input_queue = tf.train.slice_input_producer(
            [images, labels], num_epochs=num_epochs, shuffle=shuffle
        )
        
        image_l, image_ab = load_image_from_path(input_queue)
        
        
        capacity = 100 + 3 * batch_size
        
        image_batch_l, image_batch_ab = tf.train.batch(
            [image_l, image_ab],
            batch_size=batch_size,
            capacity=capacity
        )
    
    return image_batch_l, image_batch_ab
