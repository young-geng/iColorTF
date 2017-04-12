import os

import numpy as np
import skimage.color

import tensorflow as tf

from model import AttrDict


def read_image_path(image_path_file, path_prefix):
    image_path = []
    image_labels = []
    with open(image_path_file) as fin:
        for line in fin:
            ss = line.split()
            if len(ss) < 2:
                continue
            image_path.append(os.path.join(path_prefix, ss[0]))
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
    
    
def random_reveal_python(img_ab):
    mask_mult = 110.
    p_numpatch = 0.125 # probability for number of patches to use drawn from geometric distribution
    p_min_size = 0 # half-patch min size
    p_max_size = 4 # half-path max size
    p_std = .25 # std of center-gaussian to sample location from
    p_whole = 0.01 # probability of pushing through whole image
    
    # image_ab: H x W x 2
    Y = img_ab.shape[0] # image dimensions
    X = img_ab.shape[1]
    C = img_ab.shape[2] # number of input channels
    
    ret_ab = np.zeros((Y,X,C+1), dtype=np.float32) # value to return
    
    # choose number of points
    N = np.random.geometric(p=p_numpatch,size=1)
    
    # patch sizes
    Ps = np.random.random_integers(p_min_size,high=p_max_size,size=N)

    # determine patch locations, sampled from center-Gaussian
    Xs = np.clip(np.random.normal(loc=X/2.,scale=X*p_std,size=N),0,X)
    Ys = np.clip(np.random.normal(loc=Y/2.,scale=Y*p_std,size=N),0,Y)

    # whether or not to just feed in whole image
    use_whole = np.random.binomial(1,p_whole,size=1)

    if(use_whole==1): # throw in whole image
        ret_ab[:,:,:C] = img_ab
        ret_ab[:,:,-1] = mask_mult
    else: # sample points
        for nnn in xrange(N):
            p = int(Ps[nnn])
            x = int(Xs[nnn])
            y = int(Ys[nnn])

            ret_ab[y-p:y+p+1,x-p:x+p+1,:C] \
                = np.mean(np.mean(img_ab[y-p:y+p+1,x-p:x+p+1,:],axis=0),axis=0)[np.newaxis,np.newaxis,:]
            ret_ab[y-p:y+p+1,x-p:x+p+1,-1] = mask_mult
    
    return ret_ab
    
    
def random_reveal(image_ab):
    """ Randomly reveal patches in ab channels."""
    revealed = tf.py_func(random_reveal_python, [image_ab], tf.float32, stateful=False)
    revealed.set_shape([image_ab.shape[0], image_ab.shape[1], 3])
    return revealed


def read_imagenet_data(file_list_path, path_prefix, batch_size, num_epochs, shuffle=True):
    
    # Performing all the async data preprocessing on CPU.
    with tf.device('/cpu:0'):
        info = AttrDict()
        
        image_list, label_list = read_image_path(file_list_path, path_prefix)
        info.num_examples = len(image_list)
        
        images = tf.convert_to_tensor(image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
        
        input_queue = tf.train.slice_input_producer(
            [images, labels], num_epochs=num_epochs, shuffle=shuffle
        )
        
        image_l, image_ab = load_image_from_path(input_queue)
        revealed = random_reveal(image_ab)
        
        
        capacity = 100 + 3 * batch_size
        
        image_batch_l, image_batch_ab, revealed_batch = tf.train.batch(
            [image_l, image_ab, revealed],
            batch_size=batch_size,
            capacity=capacity
        )
    
    return image_batch_l, image_batch_ab, revealed_batch, info
