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


def read_imagenet_data(path, batch_size, num_epochs, shuffle=True):
    # Reads pfathes of images together with their labels
    image_list, label_list = read_image_path(path)
    
    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
    
    # Makes an input queue
    input_queue = tf.train.slice_input_producer(
        [images, labels], num_epochs=num_epochs, shuffle=shuffle
    )
    
    image_l, image_ab = load_image_from_path(input_queue)
    
    # image.set_shape([250, 250, 3])
    # image = tf.random_crop(image, [224, 224, 3])
    # image = tf.image.resize_images(image, [224, 224])
    
    # Optional Preprocessing or Data Augmentation
    # tf.image implements most of the standard image augmentation
    # image = preprocess_image(image)
    # label = preprocess_label(label)
    
    capacity = 100 + 3 * batch_size
    
    # Optional Image and Label Batching
    image_batch_l, image_batch_ab = tf.train.batch(
        [image_l, image_ab],
        batch_size=batch_size,
        capacity=capacity
    )
    
    return image_batch_l, image_batch_ab
    

def batch_rgb2lab(x):
    # x: N x D x D x 3
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