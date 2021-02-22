#!/usr/bin/env python
# coding: utf-8



import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
import pickle


with open('/mnt/disk1/project/SMhospital/capsule/ce_packages/dark_rgb.pickle', 'rb') as f:
    dark_rgb = pickle.load(f)
    
# GradCAM utilities

def ReSizeMap(cam):
    """
    resize a heatmap from 4 x 4 into 512 x 512 and normalize to a value between 0 and 1 
    cam = (4, 4) / cam_re = (512, 512)
    """
    cam_re = cv2.resize(cam, (512, 512), interpolation=cv2.INTER_CUBIC)
    cam_re /= np.max(cam_re) 

    return cam_re

def Map2RGB(cam_re):
    """
    convert the heatmap(512 x 512) to rgb image(512 x 512 x 3)
    cam_re = (512, 512) / cam_rgb = (512, 512, 3)
    """
    cam_rgba = cm.jet(cam_re) 
    cam_rgb = np.delete(cam_rgba, 3, 2)
    return cam_rgb

def DarkRGB(image, max_value = 7):
    """
    extract the dark parts of the sample image
    image = (512, 512, 3) 
    It extracts global variable only one time
    """
    r_0 = list(np.where(image.mean(axis=2) <= max_value))
    g_0 = r_0.copy()
    b_0 = r_0.copy()

    r_0.append(np.array([0 for i in range(len(r_0[0]))]))
    g_0.append(np.array([1 for i in range(len(g_0[0]))]))
    b_0.append(np.array([2 for i in range(len(b_0[0]))]))
    
    dark_rgb = [r_0, g_0, b_0]
    
    return dark_rgb

def ExcluDark(cam_rgb,dark_rgb):
    """
    exclude the heatmap within the dark parts of the image
    cam_rgb = (512, 512, 3) 
    """
    cam_rgb[tuple(dark_rgb[0])] = 0
    cam_rgb[tuple(dark_rgb[1])] = 0
    cam_rgb[tuple(dark_rgb[2])] = 0

    return cam_rgb
    
    
def OverLap(image, cam_rgb, alpha = 0.5, img_ch = 'bgr'):
    """
    overlap the heatmap on the image
    image, cam_rgb, return = (512, 512, 3) 
    """
    cam = (cam_rgb*255).astype('uint8')
    if img_ch == 'bgr':
        cam = cv2.cvtColor(cam, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted((image).astype('uint8'),1-alpha, cam, alpha,0)

def EnsemCam(nh_cam, nh_softmax, nd_cam, nd_softmax, mode = 'nor'):
    """
    combine the cam of NH model and the one of ND model on 4 x 4 heatmap level
    nh_cam, nd_cam = (4, 4) / nh_softmax, nd_softmax = (2, ) / return (4, 4)
    """
    if mode == 'nor':
        return (nh_softmax[1]*nh_cam/nh_cam.max() + nd_softmax[1]*nd_cam/nd_cam.max())
    elif mode == 'not_nor':
        return (nh_softmax[1]*nh_cam + nd_softmax[1]*nd_cam)

def EnsemProb(nh_softmax, nd_softmax):
    """
    the prob. of Ensemble model combined by the one of NH model and the other ND model
    nh_softmax, nd_softmax = (None, 2) / return (None, 2)
    """    
    softmax = np.zeros(nh_softmax.shape[:2])
    for i in range(nh_softmax.shape[0]):
        softmax[i, 0] = min(nh_softmax[i, 0], nd_softmax[i, 0])
        softmax[i, 1] = 1 - softmax[i, 0]
    return softmax

featuremap_list = ['conv2d/Relu:0', 'conv2d_1/Relu:0', 'maxp1/MaxPool:0', 
                    'conv2d_2/Relu:0', 'conv2d_3/Relu:0', 'maxp2/MaxPool:0', 
                    'conv2d_4/Relu:0', 'conv2d_5/Relu:0', 'maxp3/MaxPool:0', 
                    'conv2d_6/Relu:0', 'conv2d_7/Relu:0', 'maxp4/MaxPool:0', 
                    'conv2d_8/Relu:0', 'conv2d_9/Relu:0', 'maxp5/MaxPool:0', 
                    'conv2d_10/Relu:0', 'conv2d_11/Relu:0', 'maxp6/MaxPool:0', 
                    'conv2d_12/Relu:0', 'conv2d_13/Relu:0', 'maxp7/MaxPool:0']

class binary_model:  
    def __init__(self, model_path):
        self.test_graph = tf.Graph()
        with self.test_graph.as_default():
            self.sess = tf.compat.v1.Session(graph = self.test_graph)
            loader = tf.compat.v1.train.import_meta_graph(model_path + '.meta')
            loader.restore(self.sess, model_path)

            self.is_training = self.test_graph.get_tensor_by_name('is_training:0')
            self.x = self.test_graph.get_tensor_by_name('img:0')

            self.score = self.test_graph.get_tensor_by_name('score/BiasAdd:0')
    
    def compute_gradcam(self, image, grads, features, cam_phase = 'resized', alpha = 0.5, model_ch = 'bgr'):
        weights = np.mean(grads, axis=(0, 1))
        channel = features.shape[-1]
        cam = np.zeros(features.shape[0:2])
        for i, w in enumerate(weights):
            cam += w * features[:, :, i]
#         cam = np.maximum(cam, 0)
        cam = np.abs(cam)
#         cam /= np.max(cam)
#         cam[np.where(cam < 0.5)] = 0
        if cam_phase == 'heatmap':
            return cam 
        elif cam_phase == 'resized':
            cam_re = ReSizeMap(cam)
            cam_rgb = Map2RGB(cam_re)
            cam_rgb = ExcluDark(cam_rgb, dark_rgb)
            return cam_rgb
        elif cam_phase == 'overlap':
            cam_re = ReSizeMap(cam)
            cam_rgb = Map2RGB(cam_re)
            cam_rgb = ExcluDark(cam_rgb, dark_rgb)
            cam_final = OverLap(image, cam_rgb, alpha, img_ch = model_ch)
            return cam_final
    
    def get_softmax(self, image):
        """
        image.shape = (None, 512, 512, 3)
        """
        softmax = self.sess.run(tf.nn.softmax(self.score), feed_dict={self.x: image, self.is_training: False})
        return softmax
    
    def get_gradcam(self, image, class_idx = 1, featuremap_name = 'conv2d_11/Relu:0', desired_cam = 'resized', alpha = 0.3, model_ch = 'bgr'):
        """
        image.shape = (None, 512, 512, 3)
        """

        if featuremap_name not in featuremap_list:
            print("featuremap_name is not valid. please choose featuremap_name in this list:", featuremap_list)

        
        featuremap = self.test_graph.get_tensor_by_name(featuremap_name)
        grad = tf.gradients(self.score[:, class_idx], featuremap)[0]
        grads, features = self.sess.run([grad, featuremap], feed_dict={self.x: image, self.is_training: False})
        cams = []
        for i, img in enumerate(image):
            gradcam = self.compute_gradcam(img, grads[i], features[i], cam_phase = desired_cam, alpha = alpha, model_ch = model_ch)
            cams.append(gradcam)  
        return np.asarray(cams)
    
class ensemble_model:  
    def __init__(self, nh_path, nd_path):
        self.nh_test_graph = tf.Graph()
        self.nd_test_graph = tf.Graph()
        
        with self.nh_test_graph.as_default():
            self.nh_sess = tf.compat.v1.Session(graph = self.nh_test_graph)
            loader = tf.compat.v1.train.import_meta_graph(nh_path + '.meta')
            loader.restore(self.nh_sess, nh_path)

            self.nh_is_training = self.nh_test_graph.get_tensor_by_name('is_training:0')
            self.nh_x = self.nh_test_graph.get_tensor_by_name('img:0')
            self.nh_score = self.nh_test_graph.get_tensor_by_name('score/BiasAdd:0')
            
        with self.nd_test_graph.as_default():
            self.nd_sess = tf.compat.v1.Session(graph = self.nd_test_graph)
            loader = tf.compat.v1.train.import_meta_graph(nd_path + '.meta')
            loader.restore(self.nd_sess, nd_path)

            self.nd_is_training = self.nd_test_graph.get_tensor_by_name('is_training:0')
            self.nd_x = self.nd_test_graph.get_tensor_by_name('img:0')
            self.nd_score = self.nd_test_graph.get_tensor_by_name('score/BiasAdd:0')
    
    def compute_gradcam(self, grads, features):
        weights = np.mean(grads, axis=(0, 1))
        channel = features.shape[-1]
        cam = np.zeros(features.shape[0:2])
        for i, w in enumerate(weights):
            cam += w * features[:, :, i]
#         cam = np.maximum(cam, 0)
        cam = np.abs(cam)
#         cam[np.where(cam < 0.7)] = 0 
        return cam 

    def get_softmax(self, image):
        """
        image.shape = (None, 512, 512, 3)
        """
        nh_softmax = self.nh_sess.run(tf.nn.softmax(self.nh_score), feed_dict={self.nh_x: image, self.nh_is_training: False})
        nd_softmax = self.nd_sess.run(tf.nn.softmax(self.nd_score), feed_dict={self.nd_x: image, self.nd_is_training: False})
        
        e_softmax = EnsemProb(nh_softmax, nd_softmax)
        
        return e_softmax, nh_softmax, nd_softmax
    
    def get_gradcam(self, image, class_idx = 1, featuremap_name = 'conv2d_11/Relu:0', desired_cam = 'resized', alpha = 0.3, model_ch = 'bgr'):
        """
        image.shape = (None, 512, 512, 3)
        """
        if featuremap_name not in featuremap_list:
            print("featuremap_name is not valid. please choose featuremap_name in this list:", featuremap_list)
        nh_featuremap = self.nh_test_graph.get_tensor_by_name(featuremap_name)
        nd_featuremap = self.nd_test_graph.get_tensor_by_name(featuremap_name)
        nh_grad = tf.gradients(self.nh_score[:, class_idx], nh_featuremap)[0]
        nd_grad = tf.gradients(self.nd_score[:, class_idx], nd_featuremap)[0]
        nh_grads, nh_features, nh_softmax = self.nh_sess.run([nh_grad, nh_featuremap, tf.nn.softmax(self.nh_score)], feed_dict={self.nh_x: image, 
                                                                                                                self.nh_is_training: False})
        nd_grads, nd_features, nd_softmax = self.nd_sess.run([nd_grad, nd_featuremap, tf.nn.softmax(self.nd_score)], feed_dict={self.nd_x: image, 
                                                                                                                self.nd_is_training: False})
        cams = []
        for i, img in enumerate(image):
            nh_gradcam = self.compute_gradcam(nh_grads[i], nh_features[i])
            nd_gradcam = self.compute_gradcam(nd_grads[i], nd_features[i])
            e_gradcam = EnsemCam(nh_gradcam, nh_softmax[i], nd_gradcam, nd_softmax[i])
            if desired_cam == 'heatmap':
                cams.append(e_gradcam) 
            elif desired_cam == 'resized':
                e_gradcam = ReSizeMap(e_gradcam)
                e_gradcam = Map2RGB(e_gradcam)
                e_gradcam = ExcluDark(e_gradcam, dark_rgb)
                cams.append(e_gradcam)  
            elif desired_cam == 'overlap':
                e_gradcam = ReSizeMap(e_gradcam)
                e_gradcam = Map2RGB(e_gradcam)
                e_gradcam = ExcluDark(e_gradcam, dark_rgb)
                e_gradcam = OverLap(img, e_gradcam, alpha = alpha, img_ch = model_ch)
                cams.append(e_gradcam) 
        return np.asarray(cams)

