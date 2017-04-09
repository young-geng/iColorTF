import numpy as np

import tensorflow as tf



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        

def pad2d(input_tensor, padding):
    return tf.pad(
        input_tensor, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
        'CONSTANT'
    )
        

def conv2d(input_tensor, filters, kernel_size=3, strides=1,
           padding=1, dilation=1):
               
    if padding == 0:
        padded = input_tensor
    else:
        padded = pad2d(input_tensor, padding)
    
    return tf.layers.conv2d(
        padded,
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding='VALID', data_format='channels_last',
        dilation_rate=(dilation, dilation)
    )
    
    
def deconv2d(input_tensor, filters, kernel_size=3, strides=1):
    
    return tf.layers.conv2d_transpose(
        input_tensor,
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding='SAME', data_format='channels_last'
    )
    
    
def subsample2d(input_tensor, strides=2):
    return tf.nn.avg_pool(
        input_tensor, ksize=[1, 1, 1, 1], strides=[1, strides, strides, 1],
        padding='VALID', data_format='NHWC'
    )  # Use average pool to simulate a subsample
    

def batch_norm(input_tensor):
    return tf.layers.batch_normalization(
        input_tensor, axis=0
    )
    
    
def relu(input_tensor):
    return tf.nn.relu(input_tensor)
    
    
def conv2d_relu(*args, **kwargs):
    c = conv2d(*args, **kwargs)
    r = relu(c)
    return c, r


class iColorUNet(object):
    
    def __init__(self, data_l, data_ab, data_mask, groud_truth_ab):
    
        self.net = AttrDict()
        net = self.net
        
        net.data_l = data_l
        net.data_ab = data_ab
        net.data_mask = data_mask
        net.groud_truth_ab = groud_truth_ab
        
        self.build_unet()
        
    def build_unet(self):
        net = self.net
        
        net.data_ab_mask = tf.concat([net.data_ab, net.data_mask], axis=3)
        
        net.data_l_meansub = net.data_l - 50.0
        
        # Note here we use the caffe tradition of channel first
        net.bw_conv1_1 = conv2d(net.data_l_meansub, filters=64)
        
        net.ab_conv1_1 = conv2d(net.data_ab_mask, filters=64)
        
        net.conv1_1 = net.bw_conv1_1 + net.ab_conv1_1
        
        net.relu1_1 = relu(net.conv1_1)
        
        net.conv1_2, net.relu1_2 = conv2d_relu(net.relu1_1, filters=64)
        
        net.conv1_2norm = batch_norm(net.relu1_2)
        
        
        # Conv2
        
        net.conv1_2norm_ss = subsample2d(net.conv1_2norm)
        
        net.conv2_1, net.relu2_1 = conv2d_relu(net.conv1_2norm_ss, filters=128)
        
        net.conv2_2, net.relu2_2 = conv2d_relu(net.relu2_1, filters=128)
        
        net.conv2_2norm = batch_norm(net.relu2_2)
        
        
        # Conv3
        
        net.conv2_2norm_ss = subsample2d(net.conv2_2norm)

        net.conv3_1, net.relu3_1 = conv2d_relu(net.conv2_2norm_ss, filters=256)
        
        net.conv3_2, net.relu3_2 = conv2d_relu(net.relu3_1, filters=256)
        
        net.conv3_3, net.relu3_3 = conv2d_relu(net.relu3_2, filters=256)
        
        net.conv3_3norm = batch_norm(net.relu3_3)
        
        
        # Conv4
        
        net.conv3_3norm_ss = subsample2d(net.conv3_3norm)
        
        net.conv4_1, net.relu4_1 = conv2d_relu(net.conv3_3norm_ss, filters=512)
        
        net.conv4_2, net.relu4_2 = conv2d_relu(net.relu4_1, filters=512)
        
        net.conv4_3, net.relu4_3 = conv2d_relu(net.relu4_2, filters=512)
        
        net.conv4_3norm = batch_norm(net.relu4_3)
        
        
        # Conv 5
        
        net.conv5_1, net.relu5_1 = conv2d_relu(
            net.conv4_3norm, filters=512, padding=2, dilation=2
        )
        
        net.conv5_2, net.relu5_2 = conv2d_relu(
            net.relu5_1, filters=512, padding=2, dilation=2
        )
        
        net.conv5_3, net.relu5_3 = conv2d_relu(
            net.relu5_2, filters=512, padding=2, dilation=2
        )
        
        net.conv5_3norm = batch_norm(net.relu5_3)
        
        
        # Conv 6
        
        net.conv6_1, net.relu6_1 = conv2d_relu(
            net.conv5_3norm, filters=512, padding=2, dilation=2
        )
        
        net.conv6_2, net.relu6_2 = conv2d_relu(
            net.relu6_1, filters=512, padding=2, dilation=2
        )
        
        net.conv6_3, net.relu6_3 = conv2d_relu(
            net.relu6_2, filters=512, padding=2, dilation=2
        )
        
        net.conv6_3norm = batch_norm(net.relu6_3)
        
        
        # Conv 7
        
        net.conv7_1, net.relu7_1 = conv2d_relu(
            net.conv6_3norm, filters=512
        )
        
        net.conv7_2, net.relu7_2 = conv2d_relu(
            net.relu7_1, filters=512
        )
        
        net.conv7_3, net.relu7_3 = conv2d_relu(
            net.relu7_2, filters=512
        )
        
        net.conv7_3norm = batch_norm(net.relu7_3)
        
        
        # Conv8
        
        net.conv3_3_short = conv2d(net.conv3_3norm, filters=256)
        
        net.conv8_1 = deconv2d(
            net.conv7_3norm, filters=256, kernel_size=4, strides=2
        )
        
        net.conv8_1_comb = net.conv8_1 + net.conv3_3_short
        net.relu8_1_comb = relu(net.conv8_1_comb)
        
        net.conv8_2, net.relu8_2 = conv2d_relu(
            net.relu8_1_comb, filters=256
        )
        
        net.conv8_3, net.relu8_3 = conv2d_relu(
            net.relu8_2, filters=256
        )
        
        net.conv8_3norm = batch_norm(net.relu8_3)
        
        
        # Conv9
        
        net.conv9_1 = deconv2d(
            net.conv8_3norm, filters=128, kernel_size=4, strides=2
        )
        
        net.conv2_2_short = conv2d(
            net.conv2_2norm, filters=128
        )
        
        net.conv9_1_comb = net.conv2_2_short + net.conv9_1
        net.relu9_1_comb = relu(net.conv9_1_comb)
        
        net.conv9_2, net.relu9_2 = conv2d_relu(
            net.relu9_1_comb, filters=128
        )
        
        net.conv9_2norm = batch_norm(net.relu9_2)
        
        
        # Conv10
        
        net.conv1_2_short = conv2d(
            net.conv1_2norm, filters=128
        )
        
        net.conv10_1 = deconv2d(
            net.conv9_2norm, filters=128, kernel_size=4, strides=2
        )
        
        net.conv10_1_comb = net.conv1_2_short + net.conv10_1
        net.relu10_1_comb = relu(net.conv10_1_comb)
        
        net.conv10_2, net.relu10_2 = conv2d_relu(
            net.relu10_1_comb, filters=128
        )
        
        net.conv10_ab = conv2d(
            net.relu10_2, filters=2, kernel_size=1, padding=0
        )
        
        net.pred_ab_1 = tf.tanh(net.conv10_ab)
        
        net.pred_ab_2 = net.pred_ab_1 * 100
        
        net.loss_ab = tf.reduce_mean(
            tf.losses.absolute_difference(net.pred_ab_2, net.groud_truth_ab)
        )
        
        
    @property
    def prediction(self):
        return self.net.pred_ab_2
    
    @property
    def loss(self):
        return self.net.loss_ab
        
    
        
    
        
        
        
        
        
        