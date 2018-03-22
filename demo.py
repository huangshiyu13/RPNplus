import _init_paths
import inspect
import os
import shutil
import time
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from image_pylib import IMGLIB
from datetime import datetime  
import data_engine

VGG_MEAN = [103.939, 116.779, 123.68]

image_height = 720
image_width = 960
feature_height = int(np.ceil(image_height / 16.))
feature_width = int(np.ceil(image_width / 16.))


class RPN:
    def __init__(self, vgg16_npy_path=None, rpn_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, 'vgg16.npy')
            vgg16_npy_path = path
            print (path)
        if rpn_npy_path is None:
            exit()

        self.vgg16_params = np.load(vgg16_npy_path, encoding='latin1').item()
        self.rpn_params = np.load(rpn_npy_path, encoding='latin1').item()
        print('npy file loaded')

    def build(self, rgb):
     
        start_time = time.time()
        print('build model started')

        # Convert RGB to BGR
        red, green, blue = tf.split(rgb,3, 3)
        assert red.get_shape().as_list()[1:] == [image_height, image_width, 1]
        assert green.get_shape().as_list()[1:] == [image_height, image_width, 1]
        assert blue.get_shape().as_list()[1:] == [image_height, image_width, 1]
        bgr = tf.concat( [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ],3)
        assert bgr.get_shape().as_list()[1:] == [image_height, image_width, 3]
        # Conv layer 1
        self.conv1_1 = self.conv_layer_const(bgr, 'conv1_1')
        self.conv1_2 = self.conv_layer_const(self.conv1_1, 'conv1_2')
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        # Conv layer 2
        self.conv2_1 = self.conv_layer_const(self.pool1, 'conv2_1')
        self.conv2_2 = self.conv_layer_const(self.conv2_1, 'conv2_2')
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')


        # Conv layer 3
        self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')
        # Conv layer 4
        self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')
        # Conv layer 5
        self.conv5_1 = self.conv_layer(self.pool4, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3')

        # RPN_TEST_6(>=7)
        normalization_factor = tf.sqrt(tf.reduce_mean(tf.square(self.conv5_3)))
        self.gamma3 = tf.constant(self.rpn_params['gamma3:0'], dtype=tf.float32, name='gamma3')
        self.gamma4 = tf.constant(self.rpn_params['gamma4:0'], dtype=tf.float32, name='gamma4')
        # Pooling to the same size
        self.pool3_p = tf.nn.max_pool(self.pool3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                      name='pool3_proposal')
        # L2 Normalization
        self.pool3_p = self.pool3_p / (
            tf.sqrt(tf.reduce_mean(tf.square(self.pool3_p))) / normalization_factor) * self.gamma3
        self.pool4_p = self.pool4 / (
            tf.sqrt(tf.reduce_mean(tf.square(self.pool4))) / normalization_factor) * self.gamma4
        # Proposal Convolution

        self.conv_proposal_3 = self.conv_layer(self.pool3_p, 'conv_proposal_3', use_relu=0)
        self.relu_proposal_3 = tf.nn.relu(self.conv_proposal_3)
        self.conv_proposal_4 = self.conv_layer(self.pool4_p, 'conv_proposal_4', use_relu=0)
        self.relu_proposal_4 = tf.nn.relu(self.conv_proposal_4)
        self.conv_proposal_5 = self.conv_layer(self.conv5_3, 'conv_proposal_5', use_relu=0)
        self.relu_proposal_5 = tf.nn.relu(self.conv_proposal_5)
        # Concatrate
        self.relu_proposal_all = tf.concat([self.relu_proposal_3, self.relu_proposal_4, self.relu_proposal_5],3)
        # RPN_TEST_6(>=7)

        self.conv_cls_score = self.conv_layer(self.relu_proposal_all, 'conv_cls_score', use_relu=0)
        self.conv_bbox_pred = self.conv_layer(self.relu_proposal_all, 'conv_bbox_pred', use_relu=0)

        assert self.conv_cls_score.get_shape().as_list()[1:] == [feature_height, feature_width, 18]
        assert self.conv_bbox_pred.get_shape().as_list()[1:] == [feature_height, feature_width, 36]

        self.cls_score = tf.reshape(self.conv_cls_score, [-1, 2])
        self.bbox_pred = tf.reshape(self.conv_bbox_pred, [-1, 4])

        self.prob = tf.nn.softmax(self.cls_score, name="prob")

        self.data_dict = None
        print('build model finished: %ds' % (time.time() - start_time))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name, use_relu=1):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            if use_relu == 1:
                relu = tf.nn.relu(bias)
                return relu
            else:
                return bias

    def conv_layer_const(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter_const(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias_const(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):
        return tf.constant(self.rpn_params[name][0], name='filter')

    def get_bias(self, name):
        return tf.constant(self.rpn_params[name][1], name='biases')

    def get_conv_filter_const(self, name):
        return tf.constant(self.vgg16_params[name][0], name='filter')

    def get_bias_const(self, name):
        return tf.constant(self.vgg16_params[name][1], name='biases')


def checkFile(fileName):
    if os.path.isfile(fileName):
        return True
    else:
        print (fileName, 'is not found!')
        exit()


def checkDir(fileName, creat=False):
    if os.path.isdir(fileName):
        if creat:
            shutil.rmtree(fileName)
            os.mkdir(fileName)
    else:
        if creat:
            os.mkdir(fileName)
        else:
            print (fileName, 'is not found!')
            exit()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ('please input GPU index')
        exit()
    #gpuNow = '/gpu:'+sys.argv[1]


    modelPath = './models/model.npy'

    vggModelPath = './models/vgg16.npy'

    imageDir = './images/'
    resultsDir= './results/'
    checkDir(imageDir,False)
    checkDir(resultsDir,True)
    checkFile(vggModelPath)
    checkFile(modelPath)


    image_height = 720
    image_width = 960

    testDeal = data_engine.RPN_Test()



    
    sess = tf.Session()  
    image = tf.placeholder(tf.float32, [1, image_height, image_width, 3])

    cnn = RPN(vggModelPath, modelPath)
    with tf.name_scope('content_rpn'):
        cnn.build(image)

    imglib = IMGLIB()

    imageNames = data_engine.getAllFiles(imageDir, '.jpg')
    startTime = time.time()
    for imageName in imageNames:
        print (imageName[0])
        im = Image.open(imageName[0])
        pix = np.array(im.getdata()).reshape(1, image_height, image_width, 3).astype(np.float32)
        
        start_ = datetime.utcnow()  
        [test_prob, test_bbox_pred] = sess.run([cnn.prob, cnn.bbox_pred], feed_dict={image: pix})
        
        end_ = datetime.utcnow()  
        c = (end_ - start_)  
        print ('%s uses %d milliseconds' % (imageName[0] , c.microseconds/1000  ) )

        bbox = testDeal.rpn_nms(test_prob, test_bbox_pred)
        imglib.read_img(imageName[0])
        imglib.setBBXs(bbox, 'person')
        imglib.drawBox(0.99)
        imglib.save_img(resultsDir+'/'+imageName[1]+'.jpg')
    print ('total use time : %ds' % (time.time() - startTime))
